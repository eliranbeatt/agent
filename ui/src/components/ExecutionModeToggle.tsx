'use client';

import { ExecutionMode } from '@/types/execution';
import styles from './ExecutionModeToggle.module.css';

interface ExecutionModeToggleProps {
  mode: ExecutionMode;
  onChange: (mode: ExecutionMode) => void;
  disabled?: boolean;
}

export default function ExecutionModeToggle({ mode, onChange, disabled }: ExecutionModeToggleProps) {
  return (
    <div className={styles.container}>
      <label className={styles.label}>Execution Mode</label>
      <div className={styles.toggle}>
        <button
          className={`${styles.option} ${mode === 'predefined' ? styles.active : ''}`}
          onClick={() => onChange('predefined')}
          disabled={disabled}
        >
          <span className={styles.icon}>📋</span>
          <span className={styles.text}>Predefined Workflows</span>
        </button>
        <button
          className={`${styles.option} ${mode === 'autonomous' ? styles.active : ''}`}
          onClick={() => onChange('autonomous')}
          disabled={disabled}
        >
          <span className={styles.icon}>🤖</span>
          <span className={styles.text}>Autonomous Planning</span>
        </button>
      </div>
      <p className={styles.description}>
        {mode === 'predefined'
          ? 'Uses optimized workflows for common tasks'
          : 'Dynamically plans and spawns specialized agents'}
      </p>
    </div>
  );
}
