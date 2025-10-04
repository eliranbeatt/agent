'use client';

import { Message } from '@/types/chat';
import styles from './ChatMessage.module.css';

interface ChatMessageProps {
  message: Message;
  onSourceClick?: (sourceId: string) => void;
}

export default function ChatMessage({ message, onSourceClick }: ChatMessageProps) {
  const isUser = message.role === 'user';
  
  return (
    <div className={`${styles.message} ${isUser ? styles.userMessage : styles.assistantMessage}`}>
      <div className={styles.messageHeader}>
        <span className={styles.role}>{isUser ? 'You' : 'Assistant'}</span>
        <span className={styles.timestamp}>
          {message.timestamp.toLocaleTimeString()}
        </span>
      </div>
      <div className={styles.messageContent}>
        {message.content}
        {message.isStreaming && <span className={styles.cursor}>▊</span>}
      </div>
      {message.sources && message.sources.length > 0 && (
        <div className={styles.sources}>
          <span className={styles.sourcesLabel}>Sources:</span>
          {message.sources.map((source) => (
            <button
              key={source.id}
              className={styles.sourceChip}
              onClick={() => onSourceClick?.(source.id)}
              title={`${source.sourceFile}${source.pageNumber ? ` - Page ${source.pageNumber}` : ''}`}
            >
              {source.sourceFile}
              {source.pageNumber && ` (p.${source.pageNumber})`}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
