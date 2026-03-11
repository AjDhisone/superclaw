/**
 * GeminiAdapter — runs the agent loop via Google's Gemini API.
 *
 * Uses the Gemini REST API directly (no SDK dependency required).
 * Supports function calling for Gemini 1.5+ models.
 */
import fs from 'fs';
import path from 'path';
import {
  AgentAdapter,
  QueryOptions,
  QueryResult,
} from './agent-adapter.js';
import { executeMcpTool, getMcpToolDeclarations } from './mcp-bridge.js';

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
  console.error(`[gemini-adapter] ${message}`);
}

export class GeminiAdapter implements AgentAdapter {
  readonly name = 'Google Gemini';

  usesClaudeSDK(): boolean {
    return false;
  }

  async runQuery(options: QueryOptions): Promise<QueryResult> {
    const { prompt, containerInput, onOutput } = options;

    const apiKey = process.env.GEMINI_API_KEY || 'placeholder';
    const baseUrl =
      process.env.GEMINI_BASE_URL ||
      'https://generativelanguage.googleapis.com';
    const model = process.env.LLM_MODEL || 'gemini-2.0-flash';

    const systemPrompt = this.buildSystemPrompt(containerInput);
    const toolDeclarations = getMcpToolDeclarations();

    // Gemini uses a "contents" array with role: user | model
    const contents: Array<{
      role: string;
      parts: Array<Record<string, unknown>>;
    }> = [];

    contents.push({
      role: 'user',
      parts: [{ text: prompt }],
    });

    let closedDuringQuery = false;
    let resultText: string | null = null;

    for (let iter = 0; iter < MAX_TOOL_ITERATIONS; iter++) {
      if (shouldClose()) {
        closedDuringQuery = true;
        break;
      }

      log(`Iteration ${iter + 1}, sending to ${model}`);

      const requestBody: Record<string, unknown> = { contents };

      if (systemPrompt) {
        requestBody.systemInstruction = {
          parts: [{ text: systemPrompt }],
        };
      }

      if (toolDeclarations.length > 0) {
        requestBody.tools = [
          { functionDeclarations: toolDeclarations },
        ];
      }

      const url = `${baseUrl}/v1beta/models/${model}:generateContent?key=${apiKey}`;

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Gemini API error ${response.status}: ${errText}`);
      }

      const data = (await response.json()) as {
        candidates?: Array<{
          content?: {
            parts?: Array<{
              text?: string;
              functionCall?: { name: string; args: Record<string, unknown> };
            }>;
          };
          finishReason?: string;
        }>;
      };

      const candidate = data.candidates?.[0];
      if (!candidate?.content?.parts) break;

      // Add model response to conversation
      contents.push({
        role: 'model',
        parts: candidate.content.parts as Array<Record<string, unknown>>,
      });

      // Extract text and function calls
      const textParts: string[] = [];
      const functionCalls: Array<{
        name: string;
        args: Record<string, unknown>;
      }> = [];

      for (const part of candidate.content.parts) {
        if (part.text) textParts.push(part.text);
        if (part.functionCall) functionCalls.push(part.functionCall);
      }

      if (textParts.length > 0) {
        resultText = textParts.join('');
      }

      // No function calls → done
      if (functionCalls.length === 0) break;

      // Execute function calls and add results
      const functionResponses: Array<Record<string, unknown>> = [];

      for (const fc of functionCalls) {
        log(`Tool call: ${fc.name}(${JSON.stringify(fc.args).slice(0, 200)})`);

        let toolResult: string;
        try {
          toolResult = await executeMcpTool(
            fc.name,
            JSON.stringify(fc.args),
            containerInput,
          );
        } catch (err) {
          toolResult = `Error: ${err instanceof Error ? err.message : String(err)}`;
        }

        functionResponses.push({
          functionResponse: {
            name: fc.name,
            response: { result: toolResult },
          },
        });
      }

      contents.push({
        role: 'user',
        parts: functionResponses,
      });
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
  }): string {
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
