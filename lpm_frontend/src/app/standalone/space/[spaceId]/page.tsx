'use client';

import { Spin, message } from 'antd';
import { useParams } from 'next/navigation';

import type { SpaceInfo, SpaceMessage } from '@/service/space';
import { useSpaceStore } from '@/store/useSpaceStore';

import { useEffect, useRef, useState } from 'react';
import Markdown from '@/components/Markdown';

interface MessageGroup {
  type: 'opening' | 'discussion' | 'summary';
  messages: SpaceMessage[];
}

export default function SpaceDetail() {
  const params = useParams();
  const space_id = params.spaceId as string;
  const [loading, setLoading] = useState(true);
  const [messageGroups, setMessageGroups] = useState<MessageGroup[]>([]);
  const fetchSpaceById = useSpaceStore((state) => state.fetchSpaceById);
  const [spaceData, setSpaceData] = useState<SpaceInfo | null>(null);
  const pollingInterval = useRef<NodeJS.Timeout | null>(null);

  const channel = new BroadcastChannel('updateSpace');

  useEffect(() => {
    if (!spaceData?.status) {
      return;
    }

    channel.postMessage({ type: 'updateSpaceStatus', space_id, status: spaceData.status });
  }, [spaceData?.status]);

  // Get space details
  const fetchSpaceDetails = async () => {
    try {
      const data = await fetchSpaceById(space_id);

      if (data) {
        setSpaceData(data);

        // Group messages by type
        if (data.messages && data.messages.length > 0) {
          const groups: MessageGroup[] = [];

          // Opening messages
          const openingMessages = data.messages.filter((msg) => msg.message_type === 'opening');

          if (openingMessages.length > 0) {
            groups.push({ type: 'opening', messages: openingMessages });
          }

          // Discussion content
          const discussionMessages = data.messages.filter(
            (msg) => msg.message_type === 'discussion'
          );

          if (discussionMessages.length > 0) {
            groups.push({ type: 'discussion', messages: discussionMessages });
          }

          // Summary
          const summaryMessages = data.messages.filter((msg) => msg.message_type === 'summary');

          if (summaryMessages.length > 0) {
            groups.push({ type: 'summary', messages: summaryMessages });
          }

          setMessageGroups(groups);
        }
      }

      if (data?.status === 3 || data?.status === 4) {
        stopPolling();
      }
    } catch (error) {
      console.error('Error fetching space details:', error);
      message.error('Failed to load space details');
    } finally {
      setLoading(false);
    }
  };

  // Start polling
  const startPolling = () => {
    // Clear previous polling interval
    if (pollingInterval.current) {
      clearInterval(pollingInterval.current);
    }

    pollingInterval.current = setInterval(() => {
      fetchSpaceDetails();
    }, 1000);
  };

  const stopPolling = () => {
    if (pollingInterval.current) {
      clearInterval(pollingInterval.current);
      pollingInterval.current = null;
    }
  };

  // Fetch data and start polling when component mounts
  useEffect(() => {
    fetchSpaceDetails();
    startPolling();

    // Clear polling when component unmounts
    return () => {
      if (pollingInterval.current) {
        clearInterval(pollingInterval.current);
      }
    };
  }, [space_id]);

  // Format timestamp
  const formatTime = (timeString: string) => {
    try {
      const date = new Date(timeString);

      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return timeString;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Spin size="large" />
      </div>
    );
  }

  if (!spaceData) {
    return (
      <div className="flex items-center w-full justify-center h-screen">
        <div className="text-gray-500">Space not found or not started yet</div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto w-full">
      <div className="min-h-full flex flex-col">
        {/* Header */}
        <header className="bg-white shadow w-full max-w-[80rem] mx-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex justify-between items-start">
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <h1 className="text-3xl font-bold text-gray-900">{spaceData.title}</h1>
                  {spaceData.status === 4 ? (
                    <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      Finished
                    </span>
                  ) : spaceData.status === 3 ? (
                    <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                      Failed
                    </span>
                  ) : (
                    <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      Discussing
                    </span>
                  )}
                </div>
                <p className="text-lg text-gray-600">{spaceData.objective}</p>
              </div>
              <div className="flex flex-col items-end space-y-4">
                {/* Sort participants to ensure the host is listed first */}
                {[...spaceData.participants]
                  .sort((a, b) => {
                    if (a === spaceData.host) return -1;

                    if (b === spaceData.host) return 1;

                    return 0;
                  })
                  .map((participant, index) => {
                    // Extract name from participant URL more effectively
                    const extractNameFromUrl = (url: string): string => {
                      try {
                        // Try to extract name from URL pattern like http://43.130.9.122:5173/secondMeName/instanceId
                        const urlParts = url.split('/');

                        if (urlParts.length >= 4) {
                          return urlParts[3]; // This should be the Second Me name
                        }

                        return url.split('/').pop() || `Participant ${index}`;
                      } catch (error) {
                        console.error('Error extracting name from URL:', error);

                        return `Participant ${index}`;
                      }
                    };

                    const participantName = extractNameFromUrl(participant);
                    const isHost = participant === spaceData.host;

                    // Find role description if available
                    const participantInfo = spaceData.participants_info?.find(
                      (p) => p.url === participant
                    );
                    const roleDescription = participantInfo?.role_description || '';

                    return (
                      <div key={index} className="flex flex-col items-end">
                        <div className="flex items-center space-x-2">
                          <span className="text-sm font-medium text-gray-700">
                            {isHost ? `${participantName} (Host)` : participantName}
                          </span>
                          <img
                            alt={participantName}
                            className="w-10 h-10 rounded-full border-2 border-white"
                            src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${participant}`}
                          />
                        </div>
                        {roleDescription && (
                          <div className="mt-1 text-xs text-gray-500 max-w-[200px] text-right">
                            {roleDescription}
                          </div>
                        )}
                      </div>
                    );
                  })}
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 mx-auto py-8 pb-24 w-[80rem] max-w-full">
          {/* Messages Area */}
          <div className="bg-white rounded-lg shadow mb-8 w-full">
            <div className="px-6 py-5 border-b border-gray-200">
              <h2 className="text-lg font-medium text-gray-900">AI Collaboration Discussion</h2>
            </div>
            <div className="px-6 py-5 space-y-6">
              {messageGroups.map((group, groupIndex) => (
                <div key={groupIndex} className="space-y-4">
                  <h3 className="text-md font-medium text-gray-700 capitalize">
                    {group.type === 'opening'
                      ? 'Opening Statement'
                      : group.type === 'discussion'
                        ? 'Discussion'
                        : 'Summary'}
                  </h3>

                  {group.messages.map((_message) => (
                    <div
                      key={_message.id}
                      className="bg-white rounded-lg border border-gray-100 shadow-sm"
                    >
                      <div className="flex items-start p-4">
                        <img
                          alt="AI Avatar"
                          className="w-10 h-10 rounded-full"
                          src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${_message.sender_endpoint}`}
                        />
                        <div className="ml-4 flex-1">
                          <div className="flex items-center justify-between">
                            <div>
                              <h3 className="text-sm font-medium text-gray-900">
                                {(() => {
                                  // Extract name from participant URL more effectively
                                  const extractNameFromUrl = (url: string): string => {
                                    try {
                                      // Try to extract name from URL pattern
                                      const urlParts = url.split('/');

                                      if (urlParts.length >= 4) {
                                        return urlParts[3]; // This should be the Second Me name
                                      }

                                      return url.split('/').pop() || 'Participant';
                                    } catch (error) {
                                      console.error('Error extracting name from URL:', error);

                                      return 'Participant';
                                    }
                                  };

                                  const participantName = extractNameFromUrl(
                                    _message.sender_endpoint
                                  );

                                  return _message.role === 'host'
                                    ? `${participantName} (Host)`
                                    : participantName;
                                })()}
                              </h3>
                              <p className="text-xs text-gray-500">{_message.sender_endpoint}</p>
                            </div>
                            <span className="text-xs text-gray-500">
                              {formatTime(_message.create_time)}
                            </span>
                          </div>
                          <Markdown
                            className="!mt-2 text-sm text-gray-700 whitespace-pre-line"
                            markdownContent={_message.content}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ))}

              {messageGroups.length === 0 && (
                <div className="text-center py-10">
                  <p className="text-gray-500">
                    {spaceData.status === 3
                      ? `Oops, something went wrong.`
                      : `No messages yet. The discussion will appear here once it starts.`}
                  </p>
                </div>
              )}

              {/* Loading */}
              {![3, 4, undefined].includes(spaceData.status) && (
                <Spin className="!ml-[50%] -translate-x-1/2 !mt-10 !mb-6" size="large" />
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
