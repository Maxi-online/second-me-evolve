'use client';

import { DeleteOutlined } from '@ant-design/icons';

interface ChatSession {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: string;
}

interface ChatHistoryProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSessionClick: (sessionId: string) => void;
  onNewChat: () => void;
  onDeleteChat: (sessionId: string) => void;
}

export default function ChatHistory({
  sessions,
  activeSessionId,
  onSessionClick,
  onNewChat,
  onDeleteChat
}: ChatHistoryProps) {
  return (
    <div className="w-64 bg-gray-50 border-r h-full flex flex-col">
      <div className="p-4">
        <button
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          onClick={onNewChat}
        >
          <svg
            className="h-5 w-5"
            fill="currentColor"
            viewBox="0 0 20 20"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              clipRule="evenodd"
              d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z"
              fillRule="evenodd"
            />
          </svg>
          New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {sessions.map((session) => (
          <button
            key={session.id}
            className={`w-full text-left p-4 border-b hover:bg-gray-100 transition-colors ${
              activeSessionId === session.id ? 'bg-gray-100' : ''
            }`}
            onClick={() => onSessionClick(session.id)}
          >
            <div className="flex items-center">
              <div className="text-sm font-medium truncate">{session.title}</div>
              <div className="ml-auto text-gray-400 hover:text-gray-600 transition-colors">
                <DeleteOutlined onClick={() => onDeleteChat(session.id)} />
              </div>
            </div>
            <div className="text-xs text-gray-500 mt-1 truncate">{session.lastMessage}</div>
            <div className="text-xs text-gray-400 mt-1">{session.timestamp}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
