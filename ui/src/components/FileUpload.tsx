'use client';

import { useState, useRef, DragEvent } from 'react';
import { FileUpload as FileUploadType } from '@/types/chat';
import styles from './FileUpload.module.css';

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void;
  uploads: FileUploadType[];
}

export default function FileUpload({ onFilesSelected, uploads }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragEnter = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      onFilesSelected(files);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      onFilesSelected(files);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const getStatusIcon = (status: FileUploadType['status']) => {
    switch (status) {
      case 'uploading':
      case 'processing':
        return '⏳';
      case 'completed':
        return '✓';
      case 'error':
        return '✗';
      default:
        return '📄';
    }
  };

  return (
    <div className={styles.container}>
      <div
        className={`${styles.dropzone} ${isDragging ? styles.dragging : ''}`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.doc,.docx,.xls,.xlsx,.ppt,.pptx,.txt,.png,.jpg,.jpeg"
          onChange={handleFileSelect}
          className={styles.fileInput}
        />
        <div className={styles.dropzoneContent}>
          <span className={styles.uploadIcon}>📁</span>
          <p className={styles.dropzoneText}>
            Drag and drop files here or click to browse
          </p>
          <p className={styles.dropzoneHint}>
            Supports: PDF, Office docs, Images
          </p>
        </div>
      </div>

      {uploads.length > 0 && (
        <div className={styles.uploadsList}>
          {uploads.map((upload) => (
            <div key={upload.id} className={styles.uploadItem}>
              <span className={styles.statusIcon}>
                {getStatusIcon(upload.status)}
              </span>
              <div className={styles.uploadInfo}>
                <span className={styles.fileName}>{upload.file.name}</span>
                <div className={styles.progressBar}>
                  <div
                    className={styles.progressFill}
                    style={{ width: `${upload.progress}%` }}
                  />
                </div>
                {upload.error && (
                  <span className={styles.error}>{upload.error}</span>
                )}
              </div>
              <span className={styles.fileSize}>
                {(upload.file.size / 1024).toFixed(1)} KB
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
