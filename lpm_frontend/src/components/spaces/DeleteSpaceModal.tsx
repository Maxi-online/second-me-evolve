'use client';

import { Modal } from 'antd';
import { ExclamationCircleOutlined } from '@ant-design/icons';

interface DeleteSpaceModalProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  spaceName: string;
  loading?: boolean;
}

export default function DeleteSpaceModal({
  open,
  onClose,
  onConfirm,
  spaceName,
  loading = false
}: DeleteSpaceModalProps) {
  return (
    <Modal
      cancelText="Cancel"
      centered
      okButtonProps={{
        danger: true,
        loading: loading
      }}
      okText="Delete"
      onCancel={onClose}
      onOk={onConfirm}
      open={open}
      title={
        <div className="flex items-center gap-2 text-red-600">
          <ExclamationCircleOutlined /> Delete Space
        </div>
      }
    >
      <div className="space-y-4">
        <p>
          Are you sure you want to delete the space &quot;{spaceName}&quot;? This action cannot be
          undone.
        </p>
        <div className="bg-red-50 p-3 rounded-md border border-red-100">
          <p className="text-sm text-red-700">
            This will permanently delete the space and all its associated conversations.
          </p>
        </div>
      </div>
    </Modal>
  );
}
