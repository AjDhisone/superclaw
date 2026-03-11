/**
 * OpenAICompatibleAdapter — runs the agent loop via any OpenAI-compatible
 * endpoint (Ollama, LM Studio, vLLM, Together AI, etc.).
 *
 * Reuses the OpenAI SDK with a custom baseURL. Tool calling is attempted
 * but gracefully degrades if the model doesn't support it.
 */
import fs from 'fs';
import path from 'path';
import OpenAI from 'openai';
import {
  AgentAdapter,
  QueryOptions,
  QueryResult,
} from './agent-adapter.js';
import { executeMcpTool, getMcpToolDefinitions } from './mcp-bridge.js';

const IPC_INPUT_DIR = '/workspace/ipc/input';
const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
const MAX_TOOL_ITERATIONS = 25;

function shouldClose(): boolean {
  if (fs.existsSync(IPC_INPUT_CLOSE_SENTINEL)) {
    try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* */ }
    return true;
  }
  return false;
}

function log(message: string): void {
  console.error(`[openai-compatible-adapter] ${message}`);
}

export class OpenAICompatibleAdapter implements AgentAdapter {
  readonly name = 'OpenAI-Compatible';

  usesClaudeSDK(): boolean {
    return false;
  }

  async runQuery(options: QueryOptions): Promise<QueryResult> {
    const { prompt, containerInput, onOutput } = options;

    const baseURL = process.env.OPENAI_BASE_URL;
    if (!baseURL) {
      throw new Error(
        'OPENAI_BASE_URL is required for the openai-compatible provider',
      );
    }

    const client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY || 'not-needed',
      baseURL,
    });

    const model = process.env.LLM_MODEL || 'default';
    const systemPrompt = this.buildSystemPrompt(containerInput);
    const tools = getMcpToolDefinitions();

    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [];
    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }
    messages.push({ role: 'user', content: prompt });

    let closedDuringQuery = false;
    let resultText: string | null = null;

    for (let iter = 0; iter < MAX_TOOL_ITERATIONS; iter++) {
      if (shouldClose()) {
        closedDuringQuery = true;
        break;
      }

      log(`Iteration ${iter + 1}, sending to ${model} at ${baseURL}`);

      let response: OpenAI.Chat.Completions.ChatCompletion;
      try {
        response = await client.chat.completions.create({
          model,
          messages,
          tools: tools.length > 0 ? tools : undefined,
          stream: false,
        });
      } catch (err) {
        // If tool calling fails (e.g. model doesn't support it), retry without tools
        if (
          iter === 0 &&
          err instanceof Error &&
          (err.message.includes('tool') || err.message.includes('function'))
        ) {
          log('Tool calling not supported, retrying without tools');
          response = await client.chat.completions.create({
            model,
            messages,
            stream: false,
          });
        } else {
          throw err;
        }
      }

      const choice = response.choices[0];
      if (!choice) break;

      const assistantMsg = choice.message;
      messages.push(assistantMsg);

      if (assistantMsg.content) {
        resultText = assistantMsg.content;
      }

      if (!assistantMsg.tool_calls || assistantMsg.tool_calls.length === 0) {
        break;
      }

      for (const toolCall of assistantMsg.tool_calls) {
        const fnName = toolCall.function.name;
        const fnArgs = toolCall.function.arguments;
        log(`Tool call: ${fnName}(${fnArgs.slice(0, 200)})`);

        let toolResult: string;
        try {
          toolResult = await executeMcpTool(fnName, fnArgs, containerInput);
        } catch (err) {
          toolResult = `Error: ${err instanceof Error ? err.message : String(err)}`;
        }

        messages.push({
          role: 'tool',
          tool_call_id: toolCall.id,
          content: toolResult,
        });
      }
    }

    onOutput({
      status: 'success',
      result: resultText,
    });

    log(`Query complete. closedDuringQuery=${closedDuringQuery}`);
    return { closedDuringQuery };
  }

  private buildSystemPrompt(containerInput: { isMain: boolean }): string {
    const parts: string[] = [];
    const globalPath = '/workspace/global/CLAUDE.md';
    if (!containerInput.isMain && fs.existsSync(globalPath)) {
      parts.push(fs.readFileSync(globalPath, 'utf-8'));
    }
    const groupPath = '/workspace/group/CLAUDE.md';
    if (fs.existsSync(groupPath)) {
      parts.push(fs.readFileSync(groupPath, 'utf-8'));
    }
    return parts.join('\n\n');
  }
}
