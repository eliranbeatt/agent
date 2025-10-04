'use client';

import { useState, useRef, useEffect } from 'react';
import { Message, FileUpload as FileUploadType } from '@/types/chat';
import { ExecutionMode, ExecutionState, TaskNode, AgentInfo, ResourceUsage } from '@/types/execution';
import ChatMessage from './ChatMessage';
import FileUpload from './FileUpload';
import SourcePanel from './SourcePanel';
import ExecutionModeToggle from './ExecutionModeToggle';
import InspectorPanel from './InspectorPanel';
import styles from './ChatInterface.module.css';

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [uploads, setUploads] = useState<FileUploadType[]>([]);
  const [selectedSourceId, setSelectedSourceId] = useState<string>();
  const [showSourcePanel, setShowSourcePanel] = useState(false);
  const [showInspector, setShowInspector] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [executionMode, setExecutionMode] = useState<ExecutionMode>('predefined');
  const [executionState, setExecutionState] = useState<ExecutionState>({
    sessionId: 'session-1',
    currentStep: 0,
    maxSteps: 6,
    tokensUsed: 0,
    tokenBudget: 10000,
    status: 'idle',
    executionPath: 'predefined',
  });
  const [tasks, setTasks] = useState<TaskNode[]>([]);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [resources, setResources] = useState<ResourceUsage>({
    tokensUsed: 0,
    tokenBudget: 10000,
    stepsExecuted: 0,
    maxSteps: 6,
    activeAgents: 0,
    memoryUsage: 0,
  });
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFilesSelected = async (files: File[]) => {
    const { uploadFile } = await import('@/services/api');
    
    const newUploads: FileUploadType[] = files.map(file => ({
      id: Math.random().toString(36).substring(7),
      file,
      status: 'pending',
      progress: 0,
    }));

    setUploads(prev => [...prev, ...newUploads]);

    // Upload files to backend
    for (const upload of newUploads) {
      try {
        setUploads(prev =>
          prev.map(u => u.id === upload.id ? { ...u, status: 'uploading' } : u)
        );

        const result = await uploadFile(upload.file, (progress) => {
          setUploads(prev =>
            prev.map(u => u.id === upload.id ? { ...u, progress } : u)
          );
        });

        setUploads(prev =>
          prev.map(u => u.id === upload.id ? { 
            ...u, 
            status: 'completed', 
            progress: 100,
            fileId: result.file_id 
          } : u)
        );
      } catch (error) {
        console.error('Error uploading file:', error);
        setUploads(prev =>
          prev.map(u => u.id === upload.id ? { 
            ...u, 
            status: 'error',
            error: error instanceof Error ? error.message : 'Upload failed'
          } : u)
        );
      }
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isStreaming) return;

    const { streamChatMessage } = await import('@/services/api');

    const userMessage: Message = {
      id: Math.random().toString(36).substring(7),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    const messageContent = input.trim();
    setInput('');
    setIsStreaming(true);

    // Update execution state
    setExecutionState(prev => ({ ...prev, status: 'running', executionPath: executionMode }));

    // Create assistant message for streaming
    const assistantMessage: Message = {
      id: Math.random().toString(36).substring(7),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
    };

    setMessages(prev => [...prev, assistantMessage]);

    try {
      // Stream response from backend
      const stream = streamChatMessage({
        message: messageContent,
        session_id: executionState.sessionId,
        context: {
          execution_mode: executionMode,
          uploaded_files: uploads
            .filter(u => u.status === 'completed' && u.fileId)
            .map(u => u.fileId),
        },
      });

      for await (const event of stream) {
        if (event.type === 'progress') {
          // Update execution state with progress
          setExecutionState(prev => ({
            ...prev,
            currentStep: event.step,
            status: 'running',
          }));

          // Update resources
          setResources(prev => ({
            ...prev,
            stepsExecuted: event.step,
          }));
        } else if (event.type === 'response') {
          // Update message with final response
          setMessages(prev =>
            prev.map(msg =>
              msg.id === assistantMessage.id
                ? {
                    ...msg,
                    content: event.response,
                    isStreaming: false,
                    sources: event.sources,
                  }
                : msg
            )
          );

          // Update execution state
          setExecutionState(prev => ({
            ...prev,
            status: 'completed',
            sessionId: event.session_id,
          }));

          // Update resources
          if (event.metadata) {
            setResources(prev => ({
              ...prev,
              tokensUsed: event.metadata.tokens_used || prev.tokensUsed,
              stepsExecuted: event.metadata.steps || prev.stepsExecuted,
            }));
          }
        } else if (event.type === 'error') {
          // Handle error
          setMessages(prev =>
            prev.map(msg =>
              msg.id === assistantMessage.id
                ? {
                    ...msg,
                    content: `Error: ${event.error}`,
                    isStreaming: false,
                  }
                : msg
            )
          );

          setExecutionState(prev => ({ ...prev, status: 'error' }));
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Update message with error
      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantMessage.id
            ? {
                ...msg,
                content: `Error: ${error instanceof Error ? error.message : 'Failed to send message'}`,
                isStreaming: false,
              }
            : msg
        )
      );

      setExecutionState(prev => ({ ...prev, status: 'error' }));
    } finally {
      setIsStreaming(false);
    }
  };

  const handleStop = () => {
    setExecutionState(prev => ({ ...prev, status: 'idle' }));
    setIsStreaming(false);
  };

  const handlePause = () => {
    setExecutionState(prev => ({ ...prev, status: 'paused' }));
  };

  const handleResume = () => {
    setExecutionState(prev => ({ ...prev, status: 'running' }));
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleSourceClick = (sourceId: string) => {
    setSelectedSourceId(sourceId);
    setShowSourcePanel(true);
  };

  const allSources = messages
    .filter(msg => msg.sources)
    .flatMap(msg => msg.sources || []);

  return (
    <div className={styles.container}>
      <div className={styles.mainContent}>
        <div className={styles.header}>
          <h1 className={styles.title}>Local Agent Studio</h1>
          <div className={styles.headerButtons}>
            <button
              className={styles.inspectorButton}
              onClick={() => setShowInspector(!showInspector)}
            >
              🔍 Inspector
            </button>
            <button
              className={styles.sourcesButton}
              onClick={() => setShowSourcePanel(!showSourcePanel)}
            >
              📚 Sources ({allSources.length})
            </button>
          </div>
        </div>

        <div className={styles.messagesContainer}>
          {messages.length === 0 ? (
            <div className={styles.emptyState}>
              <span className={styles.emptyIcon}>💬</span>
              <h2>Welcome to Local Agent Studio</h2>
              <p>Upload documents and start asking questions</p>
            </div>
          ) : (
            <div className={styles.messagesList}>
              {messages.map(message => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  onSourceClick={handleSourceClick}
                />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className={styles.inputArea}>
          <ExecutionModeToggle
            mode={executionMode}
            onChange={setExecutionMode}
            disabled={isStreaming}
          />
          
          <FileUpload
            onFilesSelected={handleFilesSelected}
            uploads={uploads}
          />
          
          <div className={styles.inputContainer}>
            <textarea
              ref={inputRef}
              className={styles.input}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents..."
              rows={3}
              disabled={isStreaming}
            />
            <button
              className={styles.sendButton}
              onClick={handleSendMessage}
              disabled={!input.trim() || isStreaming}
            >
              {isStreaming ? '⏳' : '➤'}
            </button>
          </div>
        </div>
      </div>

      {showInspector && (
        <div className={styles.inspectorPanel}>
          <InspectorPanel
            executionState={executionState}
            tasks={tasks}
            agents={agents}
            resources={resources}
            onStop={handleStop}
            onPause={handlePause}
            onResume={handleResume}
            onClose={() => setShowInspector(false)}
          />
        </div>
      )}

      {showSourcePanel && (
        <div className={styles.sourcePanel}>
          <SourcePanel
            sources={allSources}
            selectedSourceId={selectedSourceId}
            onClose={() => setShowSourcePanel(false)}
          />
        </div>
      )}
    </div>
  );
}
