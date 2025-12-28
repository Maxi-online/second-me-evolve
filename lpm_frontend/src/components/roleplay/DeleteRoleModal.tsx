'use client';

import { Modal } from 'antd';

interface DeleteRoleModalProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void>;
  loading?: boolean;
}

export default function DeleteRoleModal({
  open,
  onClose,
  onConfirm,
  loading = false
}: DeleteRoleModalProps) {
  return (
    <Modal
      cancelButtonProps={{ disabled: loading }}
      cancelText="Cancel"
      centered
      okButtonProps={{ danger: true, loading }}
      okText="Delete"
      onCancel={onClose}
      onOk={onConfirm}
      open={open}
      title="Delete Role"
    >
      <p className="text-gray-600">
        Are you sure you want to delete this role? This action cannot be undone.
      </p>
    </Modal>
  );
}
