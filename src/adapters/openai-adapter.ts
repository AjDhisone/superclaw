/**
 * OpenAIAdapter — runs the agent loop via the OpenAI Chat Completions API.
 *
 * Supports tool calling (function calling) for models that have it
 * (gpt-4o, gpt-4-turbo, etc.). Falls back to plain chat for models
 * without tool support.
 *
 * This adapter does NOT use the Claude Agent SDK. It implements a
 * ReAct-style loop: send messages → receive tool calls → execute
 * tools → send results back → repeat until no more tool calls.
 *
 * The NanoClaw MCP tools (send_message, schedule_task, etc.) are
 * exposed as OpenAI function-calling tools.
 */
import fs from 'fs';
import path from 'path';
import OpenAI from 'openai';
import {
  AgentAdapter,
  QueryOptions,
  QueryResult,
  ContainerOutput,
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
  console.error(`[openai-adapter] ${message}`);
}

export class OpenAIAdapter implements AgentAdapter {
  readonly name = 'OpenAI';

  usesClaudeSDK(): boolean {
    return false;
  }

  async runQuery(options: QueryOptions): Promise<QueryResult> {
    const { prompt, containerInput, onOutput } = options;

    const client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY || 'placeholder',
      baseURL: process.env.OPENAI_BASE_URL || undefined,
    });

    const model = process.env.LLM_MODEL || 'gpt-4o';

    // Build system prompt from CLAUDE.md files (reused as generic context)
    const systemPrompt = this.buildSystemPrompt(containerInput);

    // Resolve MCP tool definitions as OpenAI function schemas
    const tools = getMcpToolDefinitions();

    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [];

    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }
    messages.push({ role: 'user', content: prompt });

    let closedDuringQuery = false;
    let resultText: string | null = null;

    // ReAct loop: call model → execute tools → repeat
    for (let iter = 0; iter < MAX_TOOL_ITERATIONS; iter++) {
      if (shouldClose()) {
        closedDuringQuery = true;
        break;
      }

      log(`Iteration ${iter + 1}, sending ${messages.length} messages to ${model}`);

      const response = await client.chat.completions.create({
        model,
        messages,
        tools: tools.length > 0 ? tools : undefined,
        stream: false,
      });

      const choice = response.choices[0];
      if (!choice) break;

      const assistantMsg = choice.message;
      messages.push(assistantMsg);

      // If the model produced text content, that's our result
      if (assistantMsg.content) {
        resultText = assistantMsg.content;
      }

      // No tool calls → we're done
      if (
        !assistantMsg.tool_calls ||
        assistantMsg.tool_calls.length === 0
      ) {
        break;
      }

      // Execute each tool call
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

  private buildSystemPrompt(containerInput: {
    isMain: boolean;
    groupFolder: string;
  }): string {
    const parts: string[] = [];

    // Global instructions
    const globalPath = '/workspace/global/CLAUDE.md';
    if (!containerInput.isMain && fs.existsSync(globalPath)) {
      parts.push(fs.readFileSync(globalPath, 'utf-8'));
    }

    // Group-specific instructions
    const groupPath = '/workspace/group/CLAUDE.md';
    if (fs.existsSync(groupPath)) {
      parts.push(fs.readFileSync(groupPath, 'utf-8'));
    }

    return parts.join('\n\n');
  }
}
