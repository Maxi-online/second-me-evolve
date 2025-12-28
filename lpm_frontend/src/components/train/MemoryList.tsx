'use client';

import { useState } from 'react';
import { Modal } from 'antd';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import 'github-markdown-css/github-markdown.css';

interface Memory {
  id: string;
  type: 'text' | 'file' | 'folder';
  name: string;
  content?: string;
  size: string;
  uploadedAt: string;
  isTrained?: boolean;
}

interface MemoryListProps {
  memories: Memory[];
  onDelete: (id: string, name: string) => void;
}

export default function MemoryList({ memories, onDelete }: MemoryListProps) {
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null);
  const [isModalVisible, setIsModalVisible] = useState(false);

  const showDetails = (record: Memory) => {
    setSelectedMemory(record);
    setIsModalVisible(true);
  };

  const getIcon = (type: string) => {
    switch (type) {
      case 'text':
        return (
          <svg
            className="w-5 h-5 text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
            />
          </svg>
        );
      case 'folder':
        return (
          <svg
            className="w-5 h-5 text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
            />
          </svg>
        );
      default:
        return (
          <svg
            className="w-5 h-5 text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
            />
          </svg>
        );
    }
  };

  return (
    <div>
      <div className="border rounded-lg overflow-hidden bg-gray-50 bg-opacity-30">
        <table className="min-w-full divide-y divide-gray-200 table-fixed">
          <thead className="bg-gray-100">
            <tr>
              <th className="w-[12%] px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">
                Type
              </th>
              <th className="w-[25%] px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">
                Name
              </th>
              <th className="w-[12%] px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">
                Size
              </th>
              <th className="w-[20%] px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">
                Uploaded
              </th>
              <th className="w-[10%] px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">
                Actions
              </th>
              <th className="w-[9%] px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">
                Details
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {memories.map((memory) => (
              <tr key={memory.id} className="hover:bg-gray-50 transition-colors duration-150">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center truncate">{getIcon(memory.type)}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap max-w-0">
                  <div className="text-sm text-gray-900 truncate" title={memory.name}>
                    {memory.name}
                  </div>
                  {/* {memory.content && (
                    <div className="text-sm text-gray-500 truncate" title={memory.content}>
                      {memory.content}
                    </div>
                  )} */}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 truncate">
                  {memory.size}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 truncate">
                  {memory.uploadedAt}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium truncate">
                  <button
                    className="text-red-600 hover:text-red-900 hover:underline focus:outline-none"
                    onClick={() => onDelete(memory.id, memory.name)}
                  >
                    Delete
                  </button>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium truncate">
                  <button
                    className="text-blue-600 hover:text-blue-900 hover:underline focus:outline-none"
                    onClick={() => showDetails(memory)}
                  >
                    Details
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Modal
        footer={null}
        onCancel={() => setIsModalVisible(false)}
        open={isModalVisible}
        title={selectedMemory?.name}
        width={800}
      >
        <div className="markdown-body p-6">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{selectedMemory?.content || ''}</ReactMarkdown>
        </div>
      </Modal>
    </div>
  );
}
