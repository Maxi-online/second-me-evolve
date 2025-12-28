'use client';

import RegisterUploadModal from '@/components/upload/RegisterUploadModal';
import { EVENT } from '@/utils/event';
import { Modal } from 'antd';
import { useEffect, useState } from 'react';

export default function ApplicationsLayout({ children }: { children: React.ReactNode }) {
  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [showPublishModal, setShowPublishModal] = useState(false);

  useEffect(() => {
    const handleShowRegister = () => {
      setShowRegisterModal(true);
    };

    addEventListener(EVENT.SHOW_REGISTER_MODAL, handleShowRegister);

    return () => {
      removeEventListener(EVENT.SHOW_REGISTER_MODAL, handleShowRegister);
    };
  }, []);

  return (
    <div className="h-full bg-secondme-warm-bg">
      {children}
      {/* Register AI Modal */}
      <Modal
        footer={[
          <button
            key="close"
            className="px-4 py-2 text-sm font-medium bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors mr-2"
            onClick={() => setShowRegisterModal(false)}
          >
            Cancel
          </button>,
          <button
            key="register"
            className="px-4 py-2 text-sm font-medium bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            onClick={() => {
              setShowRegisterModal(false);
              setShowPublishModal(true);
            }}
          >
            Go to Register
          </button>
        ]}
        onCancel={() => setShowRegisterModal(false)}
        open={showRegisterModal}
        title="AI Registration Required"
      >
        <div className="py-4">
          <p className="text-gray-600 mb-4">
            You need to register (publish) your AI before you can access this feature.
          </p>
          <p className="text-gray-600">
            Registration allows your AI to be fully activated and enables all application features.
          </p>
        </div>
      </Modal>

      {/* Publish Second Me Modal */}
      <RegisterUploadModal
        onClose={() => {
          setShowPublishModal(false);
        }}
        open={showPublishModal}
      />
    </div>
  );
}
