export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  sources?: Source[];
  isStreaming?: boolean;
}

export interface Source {
  id: string;
  content: string;
  sourceFile: string;
  pageNumber?: number;
  chunkId: string;
}

export interface FileUpload {
  id: string;
  file: File;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  error?: string;
  fileId?: string;
}
