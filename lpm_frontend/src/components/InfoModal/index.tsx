'use client';

import { useEffect, useRef } from 'react';

interface InfoModalProps {
  open: boolean;
  title: string;
  content: string | React.ReactNode;
  onClose: () => void;
}

export default function InfoModal({ open, title, content, onClose: handleClose }: InfoModalProps) {
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleEscKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && open) {
        handleClose();
      }
    };

    document.addEventListener('keydown', handleEscKey);

    return () => {
      document.removeEventListener('keydown', handleEscKey);
    };
  }, [open, handleClose]);

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
      handleClose();
    }
  };

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 bg-black/30 backdrop-blur-[2px] flex items-center justify-center p-4 z-50"
      onClick={handleBackdropClick}
    >
      <div
        ref={modalRef}
        className="bg-white rounded-xl p-6 max-w-lg w-full shadow-lg border border-gray-100"
      >
        <div className="flex justify-between items-start mb-4">
          <h3 className="text-xl font-semibold tracking-tight text-gray-900">{title}</h3>
          <button
            className="p-1.5 rounded-full text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
            onClick={handleClose}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                d="M6 18L18 6M6 6l12 12"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
              />
            </svg>
          </button>
        </div>

        <div className="prose prose-gray prose-sm max-w-none">
          {typeof content === 'string' ? (
            <p className="text-gray-600 leading-relaxed">{content}</p>
          ) : (
            content
          )}
        </div>

        <div className="mt-6 flex justify-end">
          <button
            className="px-4 py-2 text-sm font-medium bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
            onClick={handleClose}
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
}
