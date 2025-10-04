'use client';

import { useState } from 'react';
import { TaskNode, AgentInfo, ExecutionState, ResourceUsage } from '@/types/execution';
import ExecutionControls from './ExecutionControls';
import ResourceMonitor from './ResourceMonitor';
import styles from './InspectorPanel.module.css';

interface InspectorPanelProps {
  executionState: ExecutionState;
  tasks: TaskNode[];
  agents: AgentInfo[];
  resources: ResourceUsage;
  onStop: () => void;
  onPause: () => void;
  onResume: () => void;
  onClose?: () => void;
}

export default function InspectorPanel({
  executionState,
  tasks,
  agents,
  resources,
  onStop,
  onPause,
  onResume,
  onClose,
}: InspectorPanelProps) {
  const [activeTab, setActiveTab] = useState<'plan' | 'agents'>('plan');

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <h2 className={styles.title}>Execution Inspector</h2>
        {onClose && (
          <button className={styles.closeButton} onClick={onClose}>
            ✕
          </button>
        )}
      </div>

      <div className={styles.content}>
        <ExecutionControls
          executionState={executionState}
          onStop={onStop}
          onPause={onPause}
          onResume={onResume}
        />

        <ResourceMonitor resources={resources} />

        <div className={styles.tabs}>
          <button
            className={`${styles.tab} ${activeTab === 'plan' ? styles.activeTab : ''}`}
            onClick={() => setActiveTab('plan')}
          >
            📊 Plan Graph
          </button>
          <button
            className={`${styles.tab} ${activeTab === 'agents' ? styles.activeTab : ''}`}
            onClick={() => setActiveTab('agents')}
          >
            🤖 Agents ({agents.length})
          </button>
        </div>

        <div className={styles.tabContent}>
          {activeTab === 'plan' ? (
            <PlanGraph tasks={tasks} />
          ) : (
            <AgentsList agents={agents} />
          )}
        </div>
      </div>
    </div>
  );
}

function PlanGraph({ tasks }: { tasks: TaskNode[] }) {
  if (tasks.length === 0) {
    return (
      <div className={styles.emptyState}>
        <span className={styles.emptyIcon}>📋</span>
        <p>No tasks in execution plan</p>
      </div>
    );
  }

  return (
    <div className={styles.planGraph}>
      {tasks.map((task) => (
        <div key={task.id} className={styles.taskNode}>
          <div className={`${styles.taskStatus} ${styles[task.status]}`}>
            {getStatusIcon(task.status)}
          </div>
          <div className={styles.taskInfo}>
            <div className={styles.taskName}>{task.name}</div>
            {task.assignedAgent && (
              <div className={styles.taskAgent}>
                Agent: {task.assignedAgent}
              </div>
            )}
            {task.dependencies.length > 0 && (
              <div className={styles.taskDeps}>
                Depends on: {task.dependencies.length} task(s)
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function AgentsList({ agents }: { agents: AgentInfo[] }) {
  if (agents.length === 0) {
    return (
      <div className={styles.emptyState}>
        <span className={styles.emptyIcon}>🤖</span>
        <p>No agents spawned yet</p>
      </div>
    );
  }

  return (
    <div className={styles.agentsList}>
      {agents.map((agent) => (
        <div key={agent.id} className={styles.agentCard}>
          <div className={styles.agentHeader}>
            <div>
              <div className={styles.agentName}>{agent.name}</div>
              <div className={styles.agentRole}>{agent.role}</div>
            </div>
            <div className={`${styles.agentStatus} ${styles[agent.status]}`}>
              {agent.status}
            </div>
          </div>
          <div className={styles.agentStats}>
            <div className={styles.agentStat}>
              <span className={styles.agentStatLabel}>Tasks:</span>
              <span className={styles.agentStatValue}>{agent.tasksCompleted}</span>
            </div>
            <div className={styles.agentStat}>
              <span className={styles.agentStatLabel}>Tokens:</span>
              <span className={styles.agentStatValue}>{agent.tokensUsed.toLocaleString()}</span>
            </div>
          </div>
          <div className={styles.agentTime}>
            Created: {agent.createdAt.toLocaleTimeString()}
          </div>
        </div>
      ))}
    </div>
  );
}

function getStatusIcon(status: TaskNode['status']) {
  switch (status) {
    case 'pending':
      return '⏳';
    case 'running':
      return '▶';
    case 'completed':
      return '✓';
    case 'failed':
      return '✗';
  }
}
