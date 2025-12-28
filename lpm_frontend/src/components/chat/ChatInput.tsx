'use client';

import type { KeyboardEvent } from 'react';
import { useState } from 'react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
}

export default function ChatInput({ onSendMessage, disabled = false }: ChatInputProps) {
  const [message, setMessage] = useState('');

  const handleSubmit = () => {
    if (message.trim() && !disabled) {
      onSendMessage(message);
      setMessage('');
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Check if it's an Enter key from IME (Input Method Editor)
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t bg-white p-4">
      <div className="mx-auto flex gap-4">
        <textarea
          className="flex-1 resize-none rounded-lg border border-gray-200 p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={disabled}
          onChange={(e) => setMessage(e.target.value)}
          onCompositionEnd={(e) => {
            // After IME composition ends, if it's an Enter key, don't trigger send
            if (e.data === '\n') {
              e.preventDefault();
            }
          }}
          onKeyDown={handleKeyPress}
          placeholder="Message Second Me..."
          rows={1}
          value={message}
        />
        <button
          className={`px-4 py-2 rounded-lg ${
            !message.trim() || disabled
              ? 'bg-gray-100 text-gray-400'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          } transition-colors`}
          disabled={!message.trim() || disabled}
          onClick={handleSubmit}
        >
          Send
        </button>
      </div>
    </div>
  );
}
