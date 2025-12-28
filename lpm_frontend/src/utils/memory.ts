import type { Memory } from '@/app/dashboard/train/memories/page';
import type { MemoryFile } from '@/service/memory';

export const fileTransformToMemory: (file: MemoryFile) => Memory = (file: MemoryFile) => {
  return {
    id: file.id,
    type: 'file',
    name: file.name,
    content: file.raw_content,
    size: `${(file.document_size / 1024).toFixed(1)} KB`,
    uploadedAt: file.create_time,
    isTrained: file.embedding_status === 'SUCCESS'
  };
};
