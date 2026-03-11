/**
 * AnthropicAdapter — wraps the Claude Agent SDK (@anthropic-ai/claude-agent-sdk).
 *
 * This is the original NanoClaw behaviour: full Claude Code agent with
 * tool execution, session management, agent teams, MCP servers, etc.
 * Extracted here so the runner can swap it out for other providers.
 */
import fs from 'fs';
import path from 'path';
import { query, HookCallback, PreCompactHookInput } from '@anthropic-ai/claude-agent-sdk';
import {
  AgentAdapter,
  QueryOptions,
  QueryResult,
  ContainerOutput,
} from './agent-adapter.js';

// ─── Claude SDK message stream ──────────────────────────────────
interface SDKUserMessage {
  type: 'user';
  message: { role: 'user'; content: string };
  parent_tool_use_id: null;
  session_id: string;
}

class MessageStream {
  private queue: SDKUserMessage[] = [];
  private waiting: (() => void) | null = null;
  private done = false;

  push(text: string): void {
    this.queue.push({
      type: 'user',
      message: { role: 'user', content: text },
      parent_tool_use_id: null,
      session_id: '',
    });
    this.waiting?.();
  }

  end(): void {
    this.done = true;
    this.waiting?.();
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<SDKUserMessage> {
    while (true) {
      while (this.queue.length > 0) {
        yield this.queue.shift()!;
      }
      if (this.done) return;
      await new Promise<void>((r) => {
        this.waiting = r;
      });
      this.waiting = null;
    }
  }
}

// ─── IPC helpers (duplicated from index.ts to keep adapter self-contained) ──
const IPC_INPUT_DIR = '/workspace/ipc/input';
const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
const IPC_POLL_MS = 500;

function shouldClose(): boolean {
  if (fs.existsSync(IPC_INPUT_CLOSE_SENTINEL)) {
    try {
      fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL);
    } catch {
      /* ignore */
    }
    return true;
  }
  return false;
}

function drainIpcInput(): string[] {
  try {
    fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
    const files = fs
      .readdirSync(IPC_INPUT_DIR)
      .filter((f) => f.endsWith('.json'))
      .sort();

    const messages: string[] = [];
    for (const file of files) {
      const filePath = path.join(IPC_INPUT_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        fs.unlinkSync(filePath);
        if (data.type === 'message' && data.text) {
          messages.push(data.text);
        }
      } catch {
        try {
          fs.unlinkSync(filePath);
        } catch {
          /* ignore */
        }
      }
    }
    return messages;
  } catch {
    return [];
  }
}

function log(message: string): void {
  console.error(`[anthropic-adapter] ${message}`);
}

// ─── Pre-compact hook (transcript archival) ─────────────────────

interface SessionEntry {
  sessionId: string;
  fullPath: string;
  summary: string;
  firstPrompt: string;
}
interface SessionsIndex {
  entries: SessionEntry[];
}

function getSessionSummary(
  sessionId: string,
  transcriptPath: string,
): string | null {
  const projectDir = path.dirname(transcriptPath);
  const indexPath = path.join(projectDir, 'sessions-index.json');
  if (!fs.existsSync(indexPath)) return null;
  try {
    const index: SessionsIndex = JSON.parse(
      fs.readFileSync(indexPath, 'utf-8'),
    );
    return index.entries.find((e) => e.sessionId === sessionId)?.summary ?? null;
  } catch {
    return null;
  }
}

interface ParsedMessage {
  role: 'user' | 'assistant';
  content: string;
}

function parseTranscript(content: string): ParsedMessage[] {
  const messages: ParsedMessage[] = [];
  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      if (entry.type === 'user' && entry.message?.content) {
        const text =
          typeof entry.message.content === 'string'
            ? entry.message.content
            : entry.message.content
                .map((c: { text?: string }) => c.text || '')
                .join('');
        if (text) messages.push({ role: 'user', content: text });
      } else if (entry.type === 'assistant' && entry.message?.content) {
        const text = entry.message.content
          .filter((c: { type: string }) => c.type === 'text')
          .map((c: { text: string }) => c.text)
          .join('');
        if (text) messages.push({ role: 'assistant', content: text });
      }
    } catch {
      /* skip */
    }
  }
  return messages;
}

function sanitizeFilename(summary: string): string {
  return summary
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 50);
}

function generateFallbackName(): string {
  const t = new Date();
  return `conversation-${t.getHours().toString().padStart(2, '0')}${t.getMinutes().toString().padStart(2, '0')}`;
}

function formatTranscriptMarkdown(
  messages: ParsedMessage[],
  title?: string | null,
  assistantName?: string,
): string {
  const now = new Date();
  const fmt = (d: Date) =>
    d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  const lines: string[] = [
    `# ${title || 'Conversation'}`,
    '',
    `Archived: ${fmt(now)}`,
    '',
    '---',
    '',
  ];
  for (const msg of messages) {
    const sender =
      msg.role === 'user' ? 'User' : assistantName || 'Assistant';
    const content =
      msg.content.length > 2000
        ? msg.content.slice(0, 2000) + '...'
        : msg.content;
    lines.push(`**${sender}**: ${content}`, '');
  }
  return lines.join('\n');
}

function createPreCompactHook(assistantName?: string): HookCallback {
  return async (input) => {
    const preCompact = input as PreCompactHookInput;
    const transcriptPath = preCompact.transcript_path;
    const sessionId = preCompact.session_id;
    if (!transcriptPath || !fs.existsSync(transcriptPath)) return {};

    try {
      const content = fs.readFileSync(transcriptPath, 'utf-8');
      const messages = parseTranscript(content);
      if (messages.length === 0) return {};

      const summary = getSessionSummary(sessionId, transcriptPath);
      const name = summary
        ? sanitizeFilename(summary)
        : generateFallbackName();
      const conversationsDir = '/workspace/group/conversations';
      fs.mkdirSync(conversationsDir, { recursive: true });
      const date = new Date().toISOString().split('T')[0];
      const filePath = path.join(conversationsDir, `${date}-${name}.md`);
      fs.writeFileSync(
        filePath,
        formatTranscriptMarkdown(messages, summary, assistantName),
      );
      log(`Archived conversation to ${filePath}`);
    } catch (err) {
      log(
        `Failed to archive: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
    return {};
  };
}

// ─── Adapter implementation ─────────────────────────────────────

export class AnthropicAdapter implements AgentAdapter {
  readonly name = 'Anthropic (Claude Agent SDK)';

  usesClaudeSDK(): boolean {
    return true;
  }

  async runQuery(options: QueryOptions): Promise<QueryResult> {
    const {
      prompt,
      sessionId,
      mcpServerPath,
      containerInput,
      sdkEnv,
      resumeAt,
      onOutput,
    } = options;

    const stream = new MessageStream();
    stream.push(prompt);

    // Poll IPC for follow-up messages and _close sentinel
    let ipcPolling = true;
    let closedDuringQuery = false;
    const pollIpcDuringQuery = () => {
      if (!ipcPolling) return;
      if (shouldClose()) {
        log('Close sentinel detected during query, ending stream');
        closedDuringQuery = true;
        stream.end();
        ipcPolling = false;
        return;
      }
      const messages = drainIpcInput();
      for (const text of messages) {
        log(`Piping IPC message into active query (${text.length} chars)`);
        stream.push(text);
      }
      setTimeout(pollIpcDuringQuery, IPC_POLL_MS);
    };
    setTimeout(pollIpcDuringQuery, IPC_POLL_MS);

    let newSessionId: string | undefined;
    let lastAssistantUuid: string | undefined;
    let messageCount = 0;
    let resultCount = 0;

    // Load global CLAUDE.md
    const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
    let globalClaudeMd: string | undefined;
    if (!containerInput.isMain && fs.existsSync(globalClaudeMdPath)) {
      globalClaudeMd = fs.readFileSync(globalClaudeMdPath, 'utf-8');
    }

    // Discover additional directories
    const extraDirs: string[] = [];
    const extraBase = '/workspace/extra';
    if (fs.existsSync(extraBase)) {
      for (const entry of fs.readdirSync(extraBase)) {
        const fullPath = path.join(extraBase, entry);
        if (fs.statSync(fullPath).isDirectory()) {
          extraDirs.push(fullPath);
        }
      }
    }

    for await (const message of query({
      prompt: stream,
      options: {
        cwd: '/workspace/group',
        additionalDirectories: extraDirs.length > 0 ? extraDirs : undefined,
        resume: sessionId,
        resumeSessionAt: resumeAt,
        systemPrompt: globalClaudeMd
          ? {
              type: 'preset' as const,
              preset: 'claude_code' as const,
              append: globalClaudeMd,
            }
          : undefined,
        allowedTools: [
          'Bash',
          'Read',
          'Write',
          'Edit',
          'Glob',
          'Grep',
          'WebSearch',
          'WebFetch',
          'Task',
          'TaskOutput',
          'TaskStop',
          'TeamCreate',
          'TeamDelete',
          'SendMessage',
          'TodoWrite',
          'ToolSearch',
          'Skill',
          'NotebookEdit',
          'mcp__nanoclaw__*',
        ],
        env: sdkEnv,
        permissionMode: 'bypassPermissions',
        allowDangerouslySkipPermissions: true,
        settingSources: ['project', 'user'],
        mcpServers: {
          nanoclaw: {
            command: 'node',
            args: [mcpServerPath],
            env: {
              NANOCLAW_CHAT_JID: containerInput.chatJid,
              NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
              NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
            },
          },
        },
        hooks: {
          PreCompact: [
            {
              hooks: [
                createPreCompactHook(containerInput.assistantName),
              ],
            },
          ],
        },
      },
    })) {
      messageCount++;
      const msgType =
        message.type === 'system'
          ? `system/${(message as { subtype?: string }).subtype}`
          : message.type;
      log(`[msg #${messageCount}] type=${msgType}`);

      if (message.type === 'assistant' && 'uuid' in message) {
        lastAssistantUuid = (message as { uuid: string }).uuid;
      }

      if (message.type === 'system' && message.subtype === 'init') {
        newSessionId = message.session_id;
        log(`Session initialized: ${newSessionId}`);
      }

      if (
        message.type === 'system' &&
        (message as { subtype?: string }).subtype === 'task_notification'
      ) {
        const tn = message as {
          task_id: string;
          status: string;
          summary: string;
        };
        log(
          `Task notification: task=${tn.task_id} status=${tn.status} summary=${tn.summary}`,
        );
      }

      if (message.type === 'result') {
        resultCount++;
        const textResult =
          'result' in message
            ? (message as { result?: string }).result
            : null;
        log(
          `Result #${resultCount}: subtype=${message.subtype}${textResult ? ` text=${textResult.slice(0, 200)}` : ''}`,
        );
        onOutput({
          status: 'success',
          result: textResult || null,
          newSessionId,
        });
      }
    }

    ipcPolling = false;
    log(
      `Query done. Messages: ${messageCount}, results: ${resultCount}, ` +
        `lastAssistantUuid: ${lastAssistantUuid || 'none'}, closedDuringQuery: ${closedDuringQuery}`,
    );
    return { newSessionId, lastAssistantUuid, closedDuringQuery };
  }
}
