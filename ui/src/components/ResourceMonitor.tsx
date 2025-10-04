'use client';

import { ResourceUsage } from '@/types/execution';
import styles from './ResourceMonitor.module.css';

interface ResourceMonitorProps {
  resources: ResourceUsage;
}

export default function ResourceMonitor({ resources }: ResourceMonitorProps) {
  const tokenPercentage = (resources.tokensUsed / resources.tokenBudget) * 100;
  const stepPercentage = (resources.stepsExecuted / resources.maxSteps) * 100;

  return (
    <div className={styles.container}>
      <h3 className={styles.title}>Resource Usage</h3>
      
      <div className={styles.metrics}>
        <div className={styles.metric}>
          <div className={styles.metricHeader}>
            <span className={styles.metricLabel}>Tokens</span>
            <span className={styles.metricValue}>
              {resources.tokensUsed.toLocaleString()} / {resources.tokenBudget.toLocaleString()}
            </span>
          </div>
          <div className={styles.progressBar}>
            <div
              className={`${styles.progressFill} ${tokenPercentage > 80 ? styles.warning : ''}`}
              style={{ width: `${Math.min(tokenPercentage, 100)}%` }}
            />
          </div>
        </div>

        <div className={styles.metric}>
          <div className={styles.metricHeader}>
            <span className={styles.metricLabel}>Steps</span>
            <span className={styles.metricValue}>
              {resources.stepsExecuted} / {resources.maxSteps}
            </span>
          </div>
          <div className={styles.progressBar}>
            <div
              className={`${styles.progressFill} ${stepPercentage > 80 ? styles.warning : ''}`}
              style={{ width: `${Math.min(stepPercentage, 100)}%` }}
            />
          </div>
        </div>

        <div className={styles.statsGrid}>
          <div className={styles.stat}>
            <span className={styles.statIcon}>🤖</span>
            <div>
              <div className={styles.statValue}>{resources.activeAgents}</div>
              <div className={styles.statLabel}>Active Agents</div>
            </div>
          </div>
          
          <div className={styles.stat}>
            <span className={styles.statIcon}>💾</span>
            <div>
              <div className={styles.statValue}>{resources.memoryUsage.toFixed(1)} MB</div>
              <div className={styles.statLabel}>Memory</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
