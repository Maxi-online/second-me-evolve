'use client';

import { Modal } from 'antd';
import { useState, useEffect } from 'react';
import type { UpdateRoleReq } from '@/service/role';

interface EditRoleModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (data: UpdateRoleReq) => Promise<void>;
  initialData: UpdateRoleReq;
  loading?: boolean;
}

export default function EditRoleModal({
  open,
  onClose,
  onSubmit,
  initialData,
  loading = false
}: EditRoleModalProps) {
  const [form, setForm] = useState<UpdateRoleReq>(initialData);

  useEffect(() => {
    setForm(initialData);
  }, [initialData]);

  const handleSubmit = async () => {
    await onSubmit(form);
  };

  return (
    <Modal
      cancelButtonProps={{ disabled: loading }}
      cancelText="Cancel"
      centered
      okButtonProps={{ loading }}
      okText="Save"
      onCancel={onClose}
      onOk={handleSubmit}
      open={open}
      title="Edit Role"
    >
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Role Name</label>
          <input
            className="w-full px-3 py-2 border rounded-lg"
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            placeholder="e.g., Historical Figure, Professional Expert, etc."
            type="text"
            value={form.name}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Role Description</label>
          <textarea
            className="w-full px-3 py-2 border rounded-lg"
            onChange={(e) => setForm({ ...form, description: e.target.value })}
            placeholder="Describe the role's personality, background, and expertise..."
            rows={3}
            value={form.description}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">System Prompt</label>
          <textarea
            className="w-full px-3 py-2 border rounded-lg"
            onChange={(e) => setForm({ ...form, system_prompt: e.target.value })}
            placeholder="Enter the system prompt that defines how this role should behave..."
            rows={3}
            value={form.system_prompt}
          />
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <label className="font-medium">Memory Retrieval</label>
              <p className="text-sm text-gray-500">Direct and factual responses</p>
            </div>
            <button
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${form.enable_l0_retrieval ? 'bg-blue-600' : 'bg-gray-200'}`}
              onClick={() => {
                setForm({ ...form, enable_l0_retrieval: !form.enable_l0_retrieval });
              }}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${form.enable_l0_retrieval ? 'translate-x-[25px]' : 'translate-x-1'}`}
              />
            </button>
          </div>
        </div>
      </div>
    </Modal>
  );
}
