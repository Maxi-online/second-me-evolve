'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import CreateSpaceModal from '@/components/spaces/CreateSpaceModal';
import ShareSpaceModal from '@/components/spaces/ShareSpaceModal';
import DeleteSpaceModal from '@/components/spaces/DeleteSpaceModal';
import { useSpaceStore } from '@/store/useSpaceStore';
import { Spin, message } from 'antd';
import { useUploadStore } from '@/store/useUploadStore';
import type { SpaceInfo } from '@/service/space';
import { shareSpace } from '@/service/space';
import { useLoadInfoStore } from '@/store/useLoadInfoStore';
import { EVENT } from '@/utils/event';

// Mock data for demonstration - will be replaced with real data
interface spaceStatus {
  space_id: string;
  status: number;
}

export default function NetworkApps() {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showShareModal, setShowShareModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [selectedSpace, setSelectedSpace] = useState<any>(null);
  const [deleting, setDeleting] = useState(false);

  const loadInfo = useLoadInfoStore((state) => state.loadInfo);
  const isRegistered = useMemo(() => {
    return loadInfo?.status === 'online';
  }, [loadInfo]);

  const uploads = useUploadStore((state) => state.uploads);
  const currentRegisteredUpload = useMemo(() => {
    return uploads.find((upload) => upload.instance_id === loadInfo?.instance_id);
  }, [uploads, loadInfo]);

  const [shareSpaceId, setShareSpaceId] = useState<string>('');

  const channel = new BroadcastChannel('updateSpace');
  const channelDataRef = useRef<spaceStatus[]>([]);

  const spaces = useSpaceStore((state) => state.spaces);
  const loading = useSpaceStore((state) => state.loading);
  const error = useSpaceStore((state) => state.error);
  const fetchAllSpaces = useSpaceStore((state) => state.fetchAllSpaces);
  const updateSpaceStatus = useSpaceStore((state) => state.updateSpaceStatus);
  const deleteSpace = useSpaceStore((state) => state.deleteSpace);

  useEffect(() => {
    if (!isRegistered) {
      dispatchEvent(new Event(EVENT.SHOW_REGISTER_MODAL));
    }
  }, [isRegistered]);

  // Fetch spaces when component mounts - use empty dependency array to run only once
  useEffect(() => {
    console.log('Fetching spaces...');
    fetchAllSpaces();
  }, []);

  // Show error message if there's an error
  useEffect(() => {
    if (error) {
      message.error(error);
    }
  }, [error]);

  const handleCreateSpace = () => {
    // Just close the modal - the space opening is handled in CreateSpaceModal
    setShowCreateModal(false);
    fetchAllSpaces();
    // No need to open the space here as it's already done in the CreateSpaceModal component
  };

  const handleSpaceClick = (space_id: string) => {
    window.open(`/standalone/space/${space_id}`, '_blank');
  };

  channel.onmessage = (event) => {
    const data = event.data;

    if (
      data.type === 'updateSpaceStatus' &&
      data.space_id &&
      !channelDataRef.current.some((s) => s.space_id === data.space_id && s.status === data.status)
    ) {
      channelDataRef.current = channelDataRef.current.filter((s) => s.space_id !== data.space_id);
      channelDataRef.current.push({ space_id: data.space_id, status: data.status });
      updateSpaceStatus(data.space_id, data.status);
    }
  };

  // Function to format the date
  const formatDate = (dateString: string) => {
    if (!dateString) return 'Unknown date';

    try {
      const date = new Date(dateString);

      // Format to show year, month, and day
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      });
    } catch (_error: any) {
      console.error('Date formatting error:', _error.message);

      return 'Invalid date';
    }
  };

  const renderSpaceCard = (space: SpaceInfo) => {
    return (
      <div
        key={space.id}
        className="bg-white rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200 overflow-hidden cursor-pointer"
        onClick={() => handleSpaceClick(space.id)}
      >
        <div className="p-6">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">
                {space.title || 'Untitled Space'}
              </h3>
              <p className="mt-1 text-sm text-gray-600">
                {space.objective || 'No objective provided'}
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <button
                className="text-gray-400 hover:text-blue-500 transition-colors"
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedSpace(space);
                  handleShare(space.id);
                }}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                  />
                </svg>
              </button>
              <button
                className="text-gray-400 hover:text-red-500 transition-colors"
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedSpace(space);
                  setShowDeleteModal(true);
                }}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                  />
                </svg>
              </button>
              {space.status === 4 ? (
                <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                  Finished
                </span>
              ) : space.status === 3 ? (
                <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                  Failed
                </span>
              ) : (
                <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  Discussing
                </span>
              )}
            </div>
          </div>
          <div className="mb-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Participants</h4>
            <div className="flex flex-wrap gap-2">
              {Array.isArray(space.participants) && space.participants.length > 0 ? (
                // Sort participants to ensure the host is listed first
                [...space.participants]
                  .sort((a, b) => {
                    if (a === space.host) return -1;

                    if (b === space.host) return 1;

                    return 0;
                  })
                  .map((participant, index) => {
                    // Extract name from participant URL more effectively
                    const extractNameFromUrl = (url: string): string => {
                      try {
                        // Try to extract name from URL pattern
                        const urlParts = url.split('/');

                        if (urlParts.length >= 4) {
                          return urlParts[3]; // This should be the Second Me name
                        }

                        return url.split('/').pop() || `Participant ${index}`;
                      } catch (_error: any) {
                        console.error('Error extracting name from URL:', _error.message);

                        return `Participant ${index}`;
                      }
                    };

                    const participantName = extractNameFromUrl(participant);
                    const isHost = participant === space.host;

                    return (
                      <div
                        key={index}
                        className="inline-flex items-center space-x-1 px-2 py-1 rounded-full bg-blue-50 border border-blue-100"
                      >
                        <img
                          alt={participantName}
                          className="w-5 h-5 rounded-full"
                          src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${participant || index}`}
                        />
                        <span className="text-xs font-medium text-blue-800">
                          {isHost ? `${participantName} (Host)` : participantName}
                        </span>
                      </div>
                    );
                  })
              ) : (
                <div className="text-xs text-gray-500">No participants</div>
              )}
            </div>
          </div>

          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Created {formatDate(space.create_time)}</span>
            <div className="flex gap-2">
              <button
                className="text-blue-600 hover:text-blue-800 font-medium"
                onClick={(e) => {
                  e.stopPropagation();
                  handleSpaceClick(space.id);
                }}
              >
                View Space
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const handleShare = async (space_id: string) => {
    shareSpace(space_id).then((res) => {
      if (res.data.code == 0) {
        setShareSpaceId(res.data.data.space_share_id);
        setShowShareModal(true);
      } else {
        message.error(res.data.message);
      }
    });
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">My Network Apps</h1>
          <p className="mt-2 text-sm text-gray-600">
            Create a multi-AI collaboration space where multiple Second Mes work together to
            complete shared missions.
          </p>
        </div>
        <button
          className="inline-flex items-center px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          onClick={() => setShowCreateModal(true)}
        >
          <svg className="-ml-1 mr-2 h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path d="M12 4v16m8-8H4" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
          </svg>
          Create New Space
        </button>
      </div>

      {loading ? (
        <div className="flex justify-center items-center py-20">
          <Spin size="large" />
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {spaces && spaces.length > 0 ? (
            spaces.map((space) => renderSpaceCard(space))
          ) : (
            <div className="col-span-3 text-center py-10">
              <p className="text-gray-500">
                No spaces found. Create your first space to get started!
              </p>
            </div>
          )}
        </div>
      )}

      <CreateSpaceModal
        currentSecondMe={`https://app.secondme.io/${currentRegisteredUpload?.upload_name}/${currentRegisteredUpload?.instance_id}`}
        onClose={() => setShowCreateModal(false)}
        onSubmit={handleCreateSpace}
        open={showCreateModal}
      />

      <ShareSpaceModal
        isRegistered={isRegistered}
        onClose={() => {
          setShowShareModal(false);
          setSelectedSpace(null);
        }}
        open={showShareModal}
        space_id={shareSpaceId || ''}
      />

      <DeleteSpaceModal
        loading={deleting}
        onClose={() => {
          setShowDeleteModal(false);
          setSelectedSpace(null);
        }}
        onConfirm={async () => {
          setDeleting(true);

          try {
            await deleteSpace(selectedSpace.id);
            message.success('Space deleted successfully');
            setShowDeleteModal(false);
            setSelectedSpace(null);
            await fetchAllSpaces();
          } catch (_error: any) {
            message.error(_error.message || 'Failed to delete space');
          } finally {
            setDeleting(false);
          }
        }}
        open={showDeleteModal}
        spaceName={selectedSpace?.title || ''}
      />

      {/* Example section */}
      <div className="relative z-10 mt-8 text-right text-sm text-gray-500">
        <p className="text-right mb-2">Try examples:</p>
        <div className="flex gap-4 justify-end">
          <a
            className="hover:text-gray-700 hover:underline"
            href="https://app.secondme.io/example/brainstorming"
            rel="noopener noreferrer"
            target="_blank"
          >
            Brainstorming (Network)
          </a>
          <span>•</span>
          <a
            className="hover:text-gray-700 hover:underline"
            href="https://app.secondme.io/example/Icebreaker"
            rel="noopener noreferrer"
            target="_blank"
          >
            Icebreaker (Network)
          </a>
        </div>
      </div>
    </div>
  );
}
