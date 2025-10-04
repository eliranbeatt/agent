export type ExecutionMode = 'predefined' | 'autonomous';

export interface ExecutionState {
  sessionId: string;
  currentStep: number;
  maxSteps: number;
  tokensUsed: number;
  tokenBudget: number;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'error';
  executionPath: ExecutionMode;
}

export interface TaskNode {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  dependencies: string[];
  assignedAgent?: string;
}

export interface AgentInfo {
  id: string;
  name: string;
  role: string;
  status: 'active' | 'idle' | 'completed';
  tasksCompleted: number;
  tokensUsed: number;
  createdAt: Date;
}

export interface ResourceUsage {
  tokensUsed: number;
  tokenBudget: number;
  stepsExecuted: number;
  maxSteps: number;
  activeAgents: number;
  memoryUsage: number;
}
