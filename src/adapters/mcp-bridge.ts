/**
 * MCP Bridge — exposes NanoClaw MCP tools for non-Claude adapters.
 *
 * The Claude Agent SDK natively spawns ipc-mcp-stdio.ts as a subprocess.
 * For other adapters, this bridge provides the same tool definitions in
 * OpenAI function-calling format and Gemini function-declaration format,
 * plus an executeMcpTool() function that writes IPC files directly
 * (same mechanism as ipc-mcp-stdio.ts).
 */
import fs from 'fs';
import path from 'path';
import type { ContainerInput } from './agent-adapter.js';

const IPC_DIR = '/workspace/ipc';
const MESSAGES_DIR = path.join(IPC_DIR, 'messages');
const TASKS_DIR = path.join(IPC_DIR, 'tasks');

function writeIpcFile(dir: string, data: object): string {
  fs.mkdirSync(dir, { recursive: true });
  const filename = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}.json`;
  const filepath = path.join(dir, filename);
  const tempPath = `${filepath}.tmp`;
  fs.writeFileSync(tempPath, JSON.stringify(data, null, 2));
  fs.renameSync(tempPath, filepath);
  return filename;
}

// ──────────────────────────────────────────────────────────
// Tool definitions (OpenAI function-calling format)
// ──────────────────────────────────────────────────────────

interface FunctionDef {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export function getMcpToolDefinitions(): FunctionDef[] {
  return [
    {
      type: 'function',
      function: {
        name: 'send_message',
        description:
          'Send a message to the user or group immediately while you are still running.',
        parameters: {
          type: 'object',
          properties: {
            text: { type: 'string', description: 'The message text to send' },
            sender: {
              type: 'string',
              description:
                'Your role/identity name. When set, messages appear from a dedicated bot in Telegram.',
            },
          },
          required: ['text'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'schedule_task',
        description:
          'Schedule a recurring or one-time task. Returns the task ID.',
        parameters: {
          type: 'object',
          properties: {
            prompt: { type: 'string', description: 'What the agent should do when the task runs.' },
            schedule_type: {
              type: 'string',
              enum: ['cron', 'interval', 'once'],
              description: 'cron=recurring, interval=every N ms, once=run once at time',
            },
            schedule_value: {
              type: 'string',
              description: 'cron expression, interval ms, or local ISO timestamp',
            },
            context_mode: {
              type: 'string',
              enum: ['group', 'isolated'],
              description: 'group=with chat history, isolated=fresh session',
            },
          },
          required: ['prompt', 'schedule_type', 'schedule_value'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'list_tasks',
        description: 'List all scheduled tasks for this group.',
        parameters: { type: 'object', properties: {} },
      },
    },
    {
      type: 'function',
      function: {
        name: 'pause_task',
        description: 'Pause a scheduled task.',
        parameters: {
          type: 'object',
          properties: {
            task_id: { type: 'string', description: 'The task ID to pause' },
          },
          required: ['task_id'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'resume_task',
        description: 'Resume a paused task.',
        parameters: {
          type: 'object',
          properties: {
            task_id: { type: 'string', description: 'The task ID to resume' },
          },
          required: ['task_id'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'cancel_task',
        description: 'Cancel and delete a scheduled task.',
        parameters: {
          type: 'object',
          properties: {
            task_id: { type: 'string', description: 'The task ID to cancel' },
          },
          required: ['task_id'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'update_task',
        description: 'Update an existing scheduled task. Only provided fields are changed.',
        parameters: {
          type: 'object',
          properties: {
            task_id: { type: 'string', description: 'The task ID to update' },
            prompt: { type: 'string', description: 'New prompt' },
            schedule_type: {
              type: 'string',
              enum: ['cron', 'interval', 'once'],
            },
            schedule_value: { type: 'string', description: 'New schedule value' },
          },
          required: ['task_id'],
        },
      },
    },
  ];
}

// ──────────────────────────────────────────────────────────
// Tool declarations (Gemini function-declaration format)
// ──────────────────────────────────────────────────────────

interface GeminiFunctionDeclaration {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export function getMcpToolDeclarations(): GeminiFunctionDeclaration[] {
  return getMcpToolDefinitions().map((t) => ({
    name: t.function.name,
    description: t.function.description,
    parameters: t.function.parameters,
  }));
}

// ──────────────────────────────────────────────────────────
// Tool execution — mirrors ipc-mcp-stdio.ts IPC file writes
// ──────────────────────────────────────────────────────────

export async function executeMcpTool(
  name: string,
  argsJson: string,
  containerInput: ContainerInput,
): Promise<string> {
  const args = JSON.parse(argsJson) as Record<string, unknown>;
  const chatJid = containerInput.chatJid;
  const groupFolder = containerInput.groupFolder;
  const isMain = containerInput.isMain;

  switch (name) {
    case 'send_message': {
      const data = {
        type: 'message',
        chatJid,
        text: args.text as string,
        sender: (args.sender as string) || undefined,
        groupFolder,
        timestamp: new Date().toISOString(),
      };
      writeIpcFile(MESSAGES_DIR, data);
      return 'Message sent.';
    }

    case 'schedule_task': {
      const taskId = `task-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const targetJid =
        isMain && args.target_group_jid
          ? (args.target_group_jid as string)
          : chatJid;
      const data = {
        type: 'schedule_task',
        taskId,
        prompt: args.prompt as string,
        schedule_type: args.schedule_type as string,
        schedule_value: args.schedule_value as string,
        context_mode: (args.context_mode as string) || 'group',
        targetJid,
        createdBy: groupFolder,
        timestamp: new Date().toISOString(),
      };
      writeIpcFile(TASKS_DIR, data);
      return `Task ${taskId} scheduled: ${args.schedule_type} - ${args.schedule_value}`;
    }

    case 'list_tasks': {
      const tasksFile = path.join(IPC_DIR, 'current_tasks.json');
      if (!fs.existsSync(tasksFile)) return 'No scheduled tasks found.';
      const allTasks = JSON.parse(fs.readFileSync(tasksFile, 'utf-8')) as Array<{
        id: string;
        prompt: string;
        schedule_type: string;
        schedule_value: string;
        status: string;
        next_run: string;
        groupFolder: string;
      }>;
      const tasks = isMain
        ? allTasks
        : allTasks.filter((t) => t.groupFolder === groupFolder);
      if (tasks.length === 0) return 'No scheduled tasks found.';
      return (
        'Scheduled tasks:\n' +
        tasks
          .map(
            (t) =>
              `- [${t.id}] ${t.prompt.slice(0, 50)}... (${t.schedule_type}: ${t.schedule_value}) - ${t.status}, next: ${t.next_run || 'N/A'}`,
          )
          .join('\n')
      );
    }

    case 'pause_task':
    case 'resume_task':
    case 'cancel_task': {
      writeIpcFile(TASKS_DIR, {
        type: name,
        taskId: args.task_id as string,
        groupFolder,
        isMain,
        timestamp: new Date().toISOString(),
      });
      return `Task ${args.task_id} ${name.replace('_task', '')} requested.`;
    }

    case 'update_task': {
      const data: Record<string, unknown> = {
        type: 'update_task',
        taskId: args.task_id as string,
        groupFolder,
        isMain: String(isMain),
        timestamp: new Date().toISOString(),
      };
      if (args.prompt !== undefined) data.prompt = args.prompt;
      if (args.schedule_type !== undefined) data.schedule_type = args.schedule_type;
      if (args.schedule_value !== undefined) data.schedule_value = args.schedule_value;
      writeIpcFile(TASKS_DIR, data);
      return `Task ${args.task_id} update requested.`;
    }

    default:
      return `Unknown tool: ${name}`;
  }
}
