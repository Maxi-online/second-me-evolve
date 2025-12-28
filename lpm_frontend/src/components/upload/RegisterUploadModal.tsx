'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import { Modal, message, Switch, Tooltip, Typography } from 'antd';
import {
  GlobalOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  QuestionCircleOutlined
} from '@ant-design/icons';

import type { Upload } from '@/service/upload';
import { registerUpload, deleteUpload, connectUpload } from '@/service/upload';

import { useUploadStore } from '@/store/useUploadStore';
import { updateRegisteredUpload } from '@/utils/localRegisteredUpload';
import NetWorkMemberList from './NetWorkMemberList';
import { useLoadInfoStore } from '@/store/useLoadInfoStore';
import { getCurrentInfo } from '@/service/info';
import { copyToClipboard } from '@/utils/copy';

interface RegisterUploadModalProps {
  open: boolean;
  onClose: () => void;
}

export default function RegisterUploadModal({ open, onClose }: RegisterUploadModalProps) {
  const [currentUpload, setCurrentUpload] = useState<Upload | null>(null);

  const addUpload = useUploadStore((state) => state.addUpload);
  const removeUpload = useUploadStore((state) => state.removeUpload);
  const uploadsTotal = useUploadStore((state) => state.total);
  const fetchUploadList = useUploadStore((state) => state.fetchUploadList);

  const fetchLoadInfo = useLoadInfoStore((state) => state.fetchLoadInfo);
  const setLoadInfo = useLoadInfoStore((state) => state.setLoadInfo);
  const loadInfo = useLoadInfoStore((state) => state.loadInfo);

  const [registerLoading, setRegisterLoading] = useState<boolean>(false);

  const isRegistered = useMemo(() => {
    return loadInfo?.status === 'online';
  }, [loadInfo]);
  const registerStatus = useMemo(() => {
    return loadInfo?.status;
  }, [loadInfo]);

  const [messageApi, contextHolder] = message.useMessage();

  useEffect(() => {
    if (open) {
      startPolling();
      fetchUploadList();
    }
  }, [open, registerStatus]);

  useEffect(() => {
    // Clean up all polling when component closes
    if (!open && pollingIntervalsRef.current) {
      clearInterval(pollingIntervalsRef.current);
    }
  }, [open]);

  // Store polling interval IDs for each upload
  const pollingIntervalsRef = useRef<NodeJS.Timeout | null>(null);

  const startPolling = () => {
    const fetchStatus = async () => {
      try {
        const response = await getCurrentInfo();

        if (response.data.code === 0) {
          setLoadInfo(response.data.data);
          setCurrentUpload({
            upload_name: response.data.data.name,
            description: response.data.data.description,
            email: response.data.data.email,
            instance_id: response.data.data.instance_id
          });

          // If the upload is offline, we will continue and try to connect
          if (
            response.data.data.status === 'online' ||
            response.data.data.status === 'unregistered'
          ) {
            stopPolling();
          }
        } else {
          throw new Error(`Failed to poll status: ${response.data.message}`);
        }
      } catch (error) {
        console.error(`Failed to poll status:`, error);

        stopPolling();
      }
    };

    fetchStatus();

    // Create polling and save the interval ID
    const intervalId = setInterval(async () => {
      fetchStatus();
    }, 3000); // Poll every 3 seconds

    // Save reference to this polling
    if (pollingIntervalsRef.current) {
      clearInterval(pollingIntervalsRef.current);
    }

    pollingIntervalsRef.current = intervalId;

    return intervalId;
  };

  // Stop polling for a specific upload
  const stopPolling = () => {
    if (pollingIntervalsRef.current) {
      clearInterval(pollingIntervalsRef.current);
    }
  };

  // Clean up all polling when component unmounts
  useEffect(() => {
    return () => {
      // Clean up all polling
      if (pollingIntervalsRef.current) {
        clearInterval(pollingIntervalsRef.current);
      }
    };
  }, []);

  const handleConnectUpload = async () => {
    const res = await connectUpload();

    // Start polling to check upload status
    if (res.data.code === 0) {
      fetchLoadInfo();
    } else {
      messageApi.error(`failed to connect, ${res.data.message}`);
    }
  };

  const handleRegister = async () => {
    if (!currentUpload) {
      messageApi.warning('No Second Me available to register');

      return;
    }

    setRegisterLoading(true);

    try {
      const response = await registerUpload({
        upload_name: currentUpload.upload_name,
        description: currentUpload.description || '',
        email: currentUpload.email,
        instance_id: currentUpload.instance_id
      });

      if (response.data.code === 0) {
        const uploadInfo = response.data.data;

        startPolling();
        // Store upload information in the store
        addUpload(uploadInfo);
        updateRegisteredUpload({
          upload_name: uploadInfo.upload_name,
          instance_id: uploadInfo.instance_id
        });
        // Start establishing connection
        // Status === 'offline' ==> auto connect
        // handleConnectUpload();
      } else {
        messageApi.error(
          `${currentUpload.upload_name} failed to register, ${response.data.message}`
        );
      }
    } catch (error) {
      console.error('Failed to register Second Me:', error);
      messageApi.error('Failed to register Second Me');
    } finally {
      setRegisterLoading(false);
    }
  };

  const handleDelete = async (upload: Upload) => {
    if (!upload.upload_name || !upload.instance_id) {
      messageApi.error('Invalid Second Me data');

      return;
    }

    setRegisterLoading(true);

    try {
      stopPolling();
      const res = await deleteUpload();

      if (res.data.code === 0) {
        // Remove upload from store
        removeUpload(upload.instance_id);
      } else {
        throw new Error(res.data.message);
      }
    } catch {
      messageApi.error('Failed to delete Second Me');
    } finally {
      setRegisterLoading(false);
      startPolling();
    }
  };

  useEffect(() => {
    if (open && registerStatus === 'offline') {
      handleConnectUpload();
    }
  }, [open, registerStatus]);

  const renderLinkCard = () => {
    if (registerStatus === 'unregistered' || !currentUpload) return null;

    return (
      <div className="flex flex-col">
        <div className="text-sm leading-5 text-indigo-600 mb-1 font-medium">Second Me URL</div>
        <div className="flex justify-between items-center bg-indigo-50 p-4 rounded-lg shadow-sm border border-indigo-100 relative">
          <div className="text-sm text-indigo-700 truncate w-[90%]">
            https://app.secondme.io/{currentUpload.upload_name}/{currentUpload.instance_id}
          </div>
          <button
            className="absolute top-1/2 -translate-y-1/2 right-2 p-1.5 bg-indigo-100 text-indigo-600 rounded-md hover:bg-indigo-200 transition-colors"
            onClick={() => {
              copyToClipboard(
                `https://app.secondme.io/${currentUpload.upload_name}/${currentUpload.instance_id}`
              )
                .then(() => {
                  message.success('Copied.');
                })
                .catch(() => {
                  message.error('Copy failed, please copy manually.');
                });
            }}
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
              />
            </svg>
          </button>
        </div>

        <div className="flex justify-start items-center mt-4 mb-1 gap-2">
          <div className="text-sm leading-5 text-indigo-600 font-medium">Chat Endpoint</div>
          <a
            className="flex items-center"
            href="https://github.com/mindverse/Second-Me/blob/master/docs/Public%20Chat%20API.md"
            rel="noreferrer"
            target="_blank"
          >
            <QuestionCircleOutlined className="cursor-pointer text-indigo-600" />
          </a>
        </div>
        <div className="flex justify-between items-center bg-indigo-50 p-4 rounded-lg shadow-sm border border-indigo-100 relative">
          <div className="text-sm text-indigo-700 truncate w-[90%]">
            https://app.secondme.io/api/chat/{currentUpload.instance_id}/chat/completions
          </div>
          <button
            className="absolute top-1/2 -translate-y-1/2 right-2 p-1.5 bg-indigo-100 text-indigo-600 rounded-md hover:bg-indigo-200 transition-colors"
            onClick={() => {
              copyToClipboard(
                `https://app.secondme.io/api/chat/${currentUpload.instance_id}/chat/completions`
              )
                .then(() => {
                  message.success('Copied.');
                })
                .catch(() => {
                  message.error('Copy failed, please copy manually.');
                });
            }}
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
              />
            </svg>
          </button>
        </div>
      </div>
    );
  };

  return (
    <Modal
      centered
      className="register-upload-modal"
      footer={null}
      onCancel={onClose}
      open={open}
      title={
        <div className="flex items-center gap-2 text-lg">
          <GlobalOutlined className="text-blue-500" />
          <span>Join the Second Me Network</span>
        </div>
      }
      width={800}
    >
      {contextHolder}
      <div className="mb-4 px-2">
        <Typography.Paragraph className="text-gray-600">
          Connect your Second Me to the global network to interact with other digital minds. Publish
          your Second Me to make it discoverable and accessible to others.
        </Typography.Paragraph>
      </div>
      <div className="flex gap-6">
        {/* Left Side - Current Second Me Status */}
        <div className="w-3/5 space-y-4">
          <h3 className="text-base font-medium text-blue-600 border-b border-blue-200 pb-2 flex items-center gap-2">
            <ApiOutlined /> Current Second Me Status
          </h3>
          <div className="space-y-2 max-h-[450px] overflow-y-auto pr-2">
            {currentUpload ? (
              <div className="space-y-4">
                <div className="bg-blue-50 px-4 py-3 rounded-lg shadow-sm border border-blue-100">
                  <div className="flex items-center gap-2">
                    <div className="flex-shrink-0 w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white">
                      {currentUpload.upload_name
                        ? currentUpload.upload_name.charAt(0).toUpperCase()
                        : '?'}
                    </div>
                    <div className="flex-grow">
                      <div className="font-medium text-blue-700">
                        {currentUpload.upload_name || 'Unknown'}
                      </div>
                      <div className="text-xs text-blue-400">
                        {currentUpload.email || 'No email provided'}
                      </div>
                    </div>
                    {/* {renderUpdateUserInfomation()} */}
                  </div>

                  {/* Status indicator for current upload if it's registered */}
                  {registerStatus && (
                    <div className="flex items-center gap-2 mt-3">
                      <div
                        className={`px-3 py-1 text-xs rounded-full flex items-center gap-2 shadow-sm ${
                          registerStatus === 'offline'
                            ? 'bg-gray-100 text-gray-800'
                            : registerStatus === 'registered'
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-green-100 text-green-800'
                        }`}
                      >
                        {registerStatus === 'offline' ? (
                          <CloseCircleOutlined className="text-gray-500" />
                        ) : registerStatus === 'registered' ? (
                          <SyncOutlined className="text-yellow-600" spin />
                        ) : (
                          <CheckCircleOutlined className="text-green-600" />
                        )}
                        {registerStatus}
                      </div>
                    </div>
                  )}
                </div>

                {/* Publication Control */}
                <div className="flex justify-between items-center bg-indigo-50 p-4 rounded-lg shadow-sm border border-indigo-100">
                  <div className="max-w-[85%]">
                    {' '}
                    {/* Increased max-width to prevent text truncation */}
                    <p className="text-sm font-medium text-indigo-700 flex items-center gap-1.5">
                      <GlobalOutlined /> Register on the Network
                    </p>
                    <p className="text-xs text-indigo-500 mt-1.5">
                      {isRegistered
                        ? 'Your Second Me is registered and discoverable'
                        : 'Toggle to register your Second Me on the network'}
                    </p>
                  </div>
                  <Tooltip
                    title={isRegistered ? 'Unregister your Second Me' : 'Register your Second Me'}
                  >
                    <div className="transform hover:scale-105 transition-transform duration-200 ml-4">
                      {' '}
                      {/* Added left margin */}
                      <Switch
                        checked={registerStatus !== 'unregistered'}
                        className="scale-125"
                        loading={registerLoading}
                        onChange={(checked) => {
                          if (!loadInfo) {
                            return;
                          }

                          if (checked) {
                            handleRegister();
                          } else if (currentUpload.upload_name && currentUpload.instance_id) {
                            handleDelete(currentUpload);
                          }
                        }}
                      />
                    </div>
                  </Tooltip>
                </div>

                {/* Public Connection option - only shown when registered */}
                {/* {registerStatus !== 'unregistered' && (
                  <div className="flex justify-between items-center bg-green-50 p-4 rounded-lg shadow-sm border border-green-100 mt-4">
                    <div className="max-w-[85%]">
                      <p className="text-sm font-medium text-green-700 flex items-center gap-1.5">
                        <ApiOutlined /> Public Access Control
                      </p>
                      <p className="text-xs text-green-500 mt-1.5">
                        When enabled, anyone can access your Second Me and invite it to Spaces. When
                        disabled, your Second Me will not be accessible.
                      </p>
                    </div>
                    <div className="transform hover:scale-105 transition-transform duration-200 ml-4">
                      <Switch
                        className="scale-125"
                        defaultChecked={true}
                        onChange={(checked) => {
                          // This is a mock option, so we don't need to implement the backend functionality
                          message.success(`Public access ${checked ? 'enabled' : 'disabled'}`);
                        }}
                      />
                    </div>
                  </div>
                )} */}

                {renderLinkCard()}
              </div>
            ) : (
              <div className="bg-gray-50 rounded-lg border border-gray-200 p-8 text-center shadow-sm">
                <div className="flex justify-center mb-4">
                  <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center text-gray-400">
                    <CloseCircleOutlined style={{ fontSize: '24px' }} />
                  </div>
                </div>
                <p className="text-gray-500 font-medium">No Second Me Available</p>
                <p className="text-sm text-gray-400 mt-2">
                  You need to create a Second Me before joining the network
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Right Side - Registered Second Me List */}
        <div className="w-2/5 space-y-4 relative pb-8">
          <h3 className="text-base font-medium text-purple-600 border-b border-purple-200 pb-2 flex items-center gap-2">
            <GlobalOutlined /> Network Members{' '}
            <span className="bg-purple-100 text-purple-700 text-xs px-2 py-0.5 rounded-full ml-1">
              {uploadsTotal}
            </span>
          </h3>
          <div
            className="space-y-3 max-h-[380px] overflow-y-auto pr-2"
            id="netWorkMemberScrollList"
          >
            <NetWorkMemberList />
          </div>
        </div>
      </div>
    </Modal>
  );
}
