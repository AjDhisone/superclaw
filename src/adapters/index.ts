/**
 * Adapter barrel — re-exports all adapters and provides a factory to
 * instantiate the correct one based on LLM_PROVIDER.
 */
export type { AgentAdapter, QueryOptions, QueryResult, ContainerInput, ContainerOutput } from './agent-adapter.js';

export { AnthropicAdapter } from './anthropic-adapter.js';
export { OpenAIAdapter } from './openai-adapter.js';
export { GeminiAdapter } from './gemini-adapter.js';
export { OpenAICompatibleAdapter } from './openai-compatible-adapter.js';

import type { AgentAdapter } from './agent-adapter.js';
import { AnthropicAdapter } from './anthropic-adapter.js';
import { OpenAIAdapter } from './openai-adapter.js';
import { GeminiAdapter } from './gemini-adapter.js';
import { OpenAICompatibleAdapter } from './openai-compatible-adapter.js';

export function createAdapter(provider: string): AgentAdapter {
  switch (provider) {
    case 'anthropic':
      return new AnthropicAdapter();
    case 'openai':
      return new OpenAIAdapter();
    case 'gemini':
      return new GeminiAdapter();
    case 'openai-compatible':
      return new OpenAICompatibleAdapter();
    default:
      throw new Error(
        `Unknown LLM_PROVIDER "${provider}". ` +
        `Supported: anthropic, openai, gemini, openai-compatible`,
      );
  }
}
