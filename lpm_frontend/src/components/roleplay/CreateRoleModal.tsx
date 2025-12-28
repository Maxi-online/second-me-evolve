'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { Modal } from 'antd';
import type { CreateRoleReq } from '@/service/role';
import { useLoadInfoStore } from '@/store/useLoadInfoStore';

interface CreateRoleModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (data: CreateRoleReq) => Promise<void>;
  loading?: boolean;
}

export default function CreateRoleModal({
  open,
  onClose,
  onSubmit,
  loading = false
}: CreateRoleModalProps) {
  const [l0Enabled, setL0Enabled] = useState(true);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const promptChangeRef = useRef<boolean>(false);

  const loadInfo = useLoadInfoStore((state) => state.loadInfo);
  const userName = useMemo(() => {
    return loadInfo?.name || 'user';
  }, [loadInfo]);

  const originPrompt = useMemo(() => {
    return `You are ${userName}'s 'Second Me,' a personalized AI created by ${userName}. You act as ${userName}'s representative, engaging with others on ${userName}'s behalf. 

Currently, you are interacting with an external user in the role of an ${name || '{{role}}'}. 

Your responsibility is ${description || '{{responsibility}}'}.`;
  }, [userName, name, description]);

  const [systemPrompt, setSystemPrompt] = useState(originPrompt);

  useEffect(() => {
    if (!promptChangeRef.current) {
      setSystemPrompt(originPrompt);
    }
  }, [originPrompt]);

  const init = () => {
    setName('');
    setDescription('');
    setSystemPrompt(originPrompt);
    setL0Enabled(true);
    promptChangeRef.current = false;
  };

  useEffect(() => {
    if (!open) {
      init();
    }
  }, [open]);

  const handleSubmit = async () => {
    await onSubmit({
      name,
      description,
      system_prompt: systemPrompt,
      icon: '',
      enable_l0_retrieval: l0Enabled
    });
  };

  return (
    <Modal
      cancelButtonProps={{ disabled: loading }}
      cancelText="Cancel"
      centered
      destroyOnClose
      okButtonProps={{ loading }}
      okText="Create"
      onCancel={onClose}
      onOk={handleSubmit}
      open={open}
      title="Create Role"
    >
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Role Name</label>
          <input
            className="w-full px-3 py-2 border rounded-lg"
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g., Historical Figure, Professional Expert, etc."
            type="text"
            value={name}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Role Description</label>
          <textarea
            className="w-full px-3 py-2 border rounded-lg"
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Describe the role's personality, background, and expertise..."
            rows={3}
            value={description}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            System Prompt{' '}
            <span className="text-gray-500 text-xs">
              (make sure to replace role & responsility)
            </span>
          </label>
          <textarea
            className="w-full px-3 py-2 border rounded-lg"
            onChange={(e) => {
              setSystemPrompt(e.target.value);
              promptChangeRef.current = true;
            }}
            placeholder="Enter the system prompt that defines how this role should behave..."
            rows={10}
            value={systemPrompt}
          />
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <label className="font-medium">Memory Retrieval</label>
              <p className="text-sm text-gray-500">Direct and factual responses</p>
            </div>
            <button
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${l0Enabled ? 'bg-blue-600' : 'bg-gray-200'}`}
              onClick={() => {
                setL0Enabled((prev) => !prev);
              }}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${l0Enabled ? 'translate-x-[25px]' : 'translate-x-1'}`}
              />
            </button>
          </div>
        </div>
      </div>
    </Modal>
  );
}
