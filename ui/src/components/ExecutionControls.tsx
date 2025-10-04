'use client';

import { ExecutionState } from '@/types/execution';
import styles from './ExecutionControls.module.css';

interface ExecutionControlsProps {
  executionState: ExecutionState;
  onStop: () => void;
  onPause: () => void;
  onResume: () => void;
}

export default function ExecutionControls({
  executionState,
  onStop,
  onPause,
  onResume,
}: ExecutionControlsProps) {
  const { status } = executionState;
  const isRunning = status === 'running';
  const isPaused = status === 'paused';
  const canControl = isRunning || isPaused;

  return (
    <div className={styles.container}>
      <div className={styles.statusBadge}>
        <span className={`${styles.statusDot} ${styles[status]}`} />
        <span className={styles.statusText}>
          {status.charAt(0).toUpperCase() + status.slice(1)}
        </span>
      </div>

      <div className={styles.controls}>
        {isRunning && (
          <button
            className={`${styles.button} ${styles.pauseButton}`}
            onClick={onPause}
          >
            ⏸ Pause
          </button>
        )}
        
        {isPaused && (
          <button
            className={`${styles.button} ${styles.resumeButton}`}
            onClick={onResume}
          >
            ▶ Resume
          </button>
        )}
        
        {canControl && (
          <button
            className={`${styles.button} ${styles.stopButton}`}
            onClick={onStop}
          >
            ⏹ Stop
          </button>
        )}
      </div>
    </div>
  );
}
