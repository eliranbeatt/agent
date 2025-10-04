'use client';

import { Source } from '@/types/chat';
import styles from './SourcePanel.module.css';

interface SourcePanelProps {
  sources: Source[];
  selectedSourceId?: string;
  onClose?: () => void;
}

export default function SourcePanel({ sources, selectedSourceId, onClose }: SourcePanelProps) {
  const selectedSource = sources.find(s => s.id === selectedSourceId);

  if (sources.length === 0) {
    return (
      <div className={styles.panel}>
        <div className={styles.header}>
          <h3 className={styles.title}>Sources</h3>
          {onClose && (
            <button className={styles.closeButton} onClick={onClose}>
              ✕
            </button>
          )}
        </div>
        <div className={styles.emptyState}>
          <span className={styles.emptyIcon}>📚</span>
          <p>No sources available</p>
          <p className={styles.emptyHint}>
            Sources will appear here when the assistant references documents
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <h3 className={styles.title}>
          Sources ({sources.length})
        </h3>
        {onClose && (
          <button className={styles.closeButton} onClick={onClose}>
            ✕
          </button>
        )}
      </div>

      {selectedSource ? (
        <div className={styles.sourceDetail}>
          <div className={styles.sourceHeader}>
            <span className={styles.sourceFile}>{selectedSource.sourceFile}</span>
            {selectedSource.pageNumber && (
              <span className={styles.pageNumber}>Page {selectedSource.pageNumber}</span>
            )}
          </div>
          <div className={styles.sourceContent}>
            {selectedSource.content}
          </div>
          <div className={styles.sourceFooter}>
            <span className={styles.chunkId}>Chunk ID: {selectedSource.chunkId}</span>
          </div>
        </div>
      ) : (
        <div className={styles.sourcesList}>
          {sources.map((source) => (
            <div key={source.id} className={styles.sourceCard}>
              <div className={styles.sourceCardHeader}>
                <span className={styles.sourceFile}>{source.sourceFile}</span>
                {source.pageNumber && (
                  <span className={styles.pageNumber}>p.{source.pageNumber}</span>
                )}
              </div>
              <div className={styles.sourcePreview}>
                {source.content.substring(0, 150)}
                {source.content.length > 150 && '...'}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
