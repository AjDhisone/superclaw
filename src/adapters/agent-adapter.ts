/**
 * AgentAdapter — generic interface for plugging different LLM providers
 * into the NanoClaw agent runner.
 *
 * Each adapter encapsulates:
 *  - How to send a prompt to the LLM
 *  - How to handle streaming responses
 *  - How to bind and invoke tools (if supported)
 *  - How to manage conversation sessions (if supported)
 *
 * The agent runner's main loop delegates all LLM interaction to the adapter.
 */

export interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  assistantName?: string;
}

export interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

/**
 * Result from a single query iteration.
 */
export interface QueryResult {
  newSessionId?: string;
  lastAssistantUuid?: string;
  closedDuringQuery: boolean;
}

/**
 * Options for runQuery.
 */
export interface QueryOptions {
  prompt: string;
  sessionId?: string;
  mcpServerPath: string;
  containerInput: ContainerInput;
  sdkEnv: Record<string, string | undefined>;
  resumeAt?: string;
  /** Called for each streamed output chunk. */
  onOutput: (output: ContainerOutput) => void;
}

/**
 * Every LLM adapter must implement this interface.
 */
export interface AgentAdapter {
  /** Human-readable name, e.g. "Anthropic (Claude Agent SDK)" */
  readonly name: string;

  /**
   * Run a single query against the LLM.
   * The adapter is responsible for:
   *  - building provider-specific request payloads
   *  - streaming text back via `options.onOutput`
   *  - handling tool calls (if the provider supports them)
   *  - returning session metadata so the caller can resume
   */
  runQuery(options: QueryOptions): Promise<QueryResult>;

  /**
   * Return true if this adapter requires the Claude Agent SDK.
   * When true, the runner will import and call `query()` from
   * `@anthropic-ai/claude-agent-sdk` via this adapter.
   */
  usesClaudeSDK(): boolean;
}
