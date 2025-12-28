'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { Divider } from 'antd';

// interface Message {
//   id: string;
//   senderId: string;
//   content: string;
//   timestamp: string;
//   type: 'message' | 'thinking' | 'conclusion';
// }

// interface IResult {
//   id: string;
//   content: string;
//   createdAt: string;
// }

// Mock data
const getAvatarUrl = (id: string) => {
  // This should return the corresponding avatar URL based on the actual SecondMe ID
  // Currently using a simple placeholder image
  return `https://api.dicebear.com/7.x/bottts/svg?seed=${id}`;
};

const mockRoom = {
  id: '1',
  name: 'Market Analysis Room',
  objective: 'Analyze current market trends and provide strategic insights',
  participants: [
    { id: 'https://secondme.com/23581', name: 'Market Analyst' },
    { id: 'https://secondme.com/23582', name: 'Strategy Expert' }
  ],
  messages: [
    {
      id: '1',
      senderId: 'https://secondme.com/23581',
      content:
        "Based on my analysis of recent market data, I've observed a significant shift in consumer behavior towards sustainable products. The trend is particularly strong in the 25-40 age demographic.",
      timestamp: '5 minutes ago',
      type: 'message'
    },
    {
      id: '2',
      senderId: 'https://secondme.com/23581',
      content: 'Analyzing purchase patterns and social media sentiment...',
      timestamp: '4 minutes ago',
      type: 'thinking'
    },
    {
      id: '3',
      senderId: 'https://secondme.com/23582',
      content:
        "That aligns with the global trends I've been tracking. Looking at comparable markets in Europe and Asia, we're seeing similar patterns. Let me analyze the potential market size...",
      timestamp: '3 minutes ago',
      type: 'message'
    },
    {
      id: '4',
      senderId: 'https://secondme.com/23582',
      content: 'Calculating market size and growth projections...',
      timestamp: '2 minutes ago',
      type: 'thinking'
    },
    {
      id: '5',
      senderId: 'https://secondme.com/23581',
      content:
        'Based on our combined analysis, I suggest we focus on three key areas: eco-friendly packaging, sustainable materials, and energy-efficient products. The total addressable market for these categories is projected to reach $500B by 2025.',
      timestamp: '1 minute ago',
      type: 'conclusion'
    }
  ],
  results: [
    {
      id: '1',
      content:
        'Identified key market opportunities in sustainable product categories with specific actionable recommendations.',
      createdAt: '1 minute ago'
    }
  ]
};

export default function RoomDetail() {
  const params = useParams();
  const [room, _setRoom] = useState(mockRoom);

  // In a real app, we would fetch the room data here
  useEffect(() => {
    // fetchRoomData(params.roomId)
  }, [params.roomId]);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-3xl font-bold text-gray-900">{room.name}</h1>
          <p className="mt-2 text-lg text-gray-600">{room.objective}</p>
          <div className="flex items-center mt-4 space-x-2">
            <span className="text-sm text-gray-500">Participants:</span>
            {room.participants.map((participant) => (
              <span
                key={participant.id}
                className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800"
              >
                {participant.name}
              </span>
            ))}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-lg shadow">
          {/* Communication Area */}
          <div className="px-6 py-6">
            <h2 className="text-xl font-semibold mb-6">Communication Process</h2>
            <div className="space-y-6">
              {room.messages.map((message) => {
                const sender = room.participants.find((p) => p.id === message.senderId);
                const isThinking = message.type === 'thinking';
                const isConclusion = message.type === 'conclusion';

                return (
                  <div
                    key={message.id}
                    className={`flex space-x-4 ${isThinking ? 'opacity-70' : ''} ${isConclusion ? 'bg-blue-50 p-4 rounded-lg border border-blue-100' : ''}`}
                  >
                    <div className="flex-shrink-0 w-10 h-10">
                      <img
                        alt={sender?.name}
                        className="w-10 h-10 rounded-full"
                        src={getAvatarUrl(message.senderId)}
                      />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="text-sm font-medium text-gray-900">{sender?.name}</span>
                        <span className="text-xs text-gray-500">{message.timestamp}</span>
                        {isThinking && (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800">
                            Thinking...
                          </span>
                        )}
                        {isConclusion && (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800">
                            Conclusion
                          </span>
                        )}
                      </div>
                      <div className={`text-gray-700 ${isThinking ? 'italic' : ''}`}>
                        {message.content}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <Divider style={{ margin: '0' }} />

          {/* Results Area */}
          <div className="bg-gray-50 px-6 py-6 rounded-b-lg">
            <h2 className="text-xl font-semibold mb-6">Results</h2>
            <div className="space-y-4">
              {room.results.map((result) => (
                <div key={result.id} className="bg-white rounded-lg border border-gray-200 p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-900">Summary</span>
                    <span className="text-xs text-gray-500">{result.createdAt}</span>
                  </div>
                  <p className="text-gray-700">{result.content}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
