'use client';

import { copyToClipboard } from '@/utils/copy';
import { Modal, message } from 'antd';
import { useEffect, useState } from 'react';

interface ShareSpaceModalProps {
  open: boolean;
  onClose: () => void;
  space_id: string;
  isRegistered: boolean;
}

export default function ShareSpaceModal({
  open,
  onClose,
  space_id,
  isRegistered
}: ShareSpaceModalProps) {
  const [messageApi, contextHolder] = message.useMessage();
  const [shareUrl, setShareUrl] = useState('');

  useEffect(() => {
    if (isRegistered) {
      const secondMe = JSON.parse(localStorage.getItem('registeredUpload') || '{}');

      // TODO Later replace with IP address returned from backend
      setShareUrl(
        `https://app.secondme.io/space/${secondMe.upload_name}/${secondMe.instance_id}/${space_id}`
      );
    }
  }, [isRegistered, space_id]);

  const handleCopyLink = async () => {
    copyToClipboard(shareUrl)
      .then(() => {
        messageApi.success({
          content: 'Link copied.'
        });
      })
      .catch(() => {
        messageApi.error({
          content: 'Copy failed, please copy manually.'
        });
      });
  };

  return (
    <>
      {contextHolder}
      <Modal centered footer={null} onCancel={onClose} open={open} title="Share Space">
        {isRegistered ? (
          <div className="space-y-4">
            <input
              className="w-full px-3 py-2 border rounded-lg bg-gray-50"
              readOnly
              value={shareUrl}
            />
            <button
              className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
              onClick={handleCopyLink}
            >
              Copy Link
            </button>
          </div>
        ) : (
          <div className="text-center text-red-500">
            Please register this Second Me before sharing
          </div>
        )}
      </Modal>
    </>
  );
}
