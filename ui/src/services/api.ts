/**
 * API service for communicating with the FastAPI backend.
 * Handles all HTTP requests and WebSocket connections with proper error handling.
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ChatRequest {
  message: string;
  session_id?: string;
  context?: Record<string, any>;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  sources: Array<{
    chunk_id: string;
    content: string;
    source_file: string;
    page_number?: number;
  }>;
  execution_path: string;
  metadata: Record<string, any>;
}

export interface FileUploadResponse {
  file_id: string;
  filename: string;
  status: string;
  message: string;
}

export interface MemoryEntry {
  id: string;
  type: string;
  content: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

export interface ConfigUpdate {
  section: string;
  key: string;
  value: any;
}

/**
 * Send a chat message and get a response
 */
export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/chat/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to send message');
    }

    return await response.json();
  } catch (error) {
    console.error('Error sending chat message:', error);
    throw error;
  }
}

/**
 * Stream a chat response using Server-Sent Events
 */
export async function* streamChatMessage(request: ChatRequest): AsyncGenerator<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error('Failed to start streaming');
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error('No response body');
    }

    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data.trim()) {
            try {
              yield JSON.parse(data);
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }
    }
  } catch (error) {
    console.error('Error streaming chat message:', error);
    throw error;
  }
}

/**
 * Upload a file for processing
 */
export async function uploadFile(
  file: File,
  onProgress?: (progress: number) => void
): Promise<FileUploadResponse> {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const xhr = new XMLHttpRequest();

    return new Promise((resolve, reject) => {
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable && onProgress) {
          const progress = (e.loaded / e.total) * 100;
          onProgress(progress);
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(new Error('Invalid response format'));
          }
        } else {
          reject(new Error(`Upload failed: ${xhr.statusText}`));
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('Network error during upload'));
      });

      xhr.open('POST', `${API_BASE_URL}/files/upload`);
      xhr.send(formData);
    });
  } catch (error) {
    console.error('Error uploading file:', error);
    throw error;
  }
}

/**
 * Get file processing status
 */
export async function getFileStatus(fileId: string): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/files/${fileId}/status`);

    if (!response.ok) {
      throw new Error('Failed to get file status');
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting file status:', error);
    throw error;
  }
}

/**
 * Get memory entries for a session
 */
export async function getMemory(sessionId: string): Promise<MemoryEntry[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/memory/${sessionId}`);

    if (!response.ok) {
      throw new Error('Failed to get memory');
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting memory:', error);
    throw error;
  }
}

/**
 * Update memory entry
 */
export async function updateMemory(
  sessionId: string,
  entryId: string,
  content: string
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/memory/${sessionId}/${entryId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ content }),
    });

    if (!response.ok) {
      throw new Error('Failed to update memory');
    }
  } catch (error) {
    console.error('Error updating memory:', error);
    throw error;
  }
}

/**
 * Get current configuration
 */
export async function getConfig(): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/config/`);

    if (!response.ok) {
      throw new Error('Failed to get configuration');
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting configuration:', error);
    throw error;
  }
}

/**
 * Update configuration
 */
export async function updateConfig(update: ConfigUpdate): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/config/`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(update),
    });

    if (!response.ok) {
      throw new Error('Failed to update configuration');
    }
  } catch (error) {
    console.error('Error updating configuration:', error);
    throw error;
  }
}

/**
 * Health check
 */
export async function healthCheck(): Promise<{ status: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/healthz`);

    if (!response.ok) {
      throw new Error('Health check failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Error during health check:', error);
    throw error;
  }
}

/**
 * WebSocket connection for real-time updates
 */
export class WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor(
    private sessionId: string,
    private onMessage: (data: any) => void,
    private onError?: (error: Event) => void,
    private onClose?: () => void
  ) {}

  connect(): void {
    const wsUrl = API_BASE_URL.replace('http', 'ws');
    this.ws = new WebSocket(`${wsUrl}/ws/${this.sessionId}`);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.onMessage(data);
      } catch (e) {
        console.error('Error parsing WebSocket message:', e);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (this.onError) {
        this.onError(error);
      }
    };

    this.ws.onclose = () => {
      console.log('WebSocket closed');
      if (this.onClose) {
        this.onClose();
      }
      this.attemptReconnect();
    };
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      setTimeout(() => this.connect(), this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('Max reconnection attempts reached');
    }
  }

  send(data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.error('WebSocket is not connected');
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
