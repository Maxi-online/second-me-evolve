'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import type { RoleRes, UpdateRoleReq } from '@/service/role';
import { Modal, message } from 'antd';
import { MoreOutlined } from '@ant-design/icons';
import { useLoadInfoStore } from '@/store/useLoadInfoStore';
import { copyToClipboard } from '@/utils/copy';

interface RoleCardProps {
  role: RoleRes;
  onClick: () => void;
  onEdit: (role_id: number, data: UpdateRoleReq) => Promise<void>;
  onDelete: (role_id: number) => Promise<void>;
}

export default function RoleCard({ role, onClick, onEdit, onDelete }: RoleCardProps) {
  const [showMenu, setShowMenu] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showShareModal, setShowShareModal] = useState(false);

  const loadInfo = useLoadInfoStore((state) => state.loadInfo);
  const isRegistered = useMemo(() => {
    return loadInfo?.status === 'online';
  }, [loadInfo]);

  const menuRef = useRef<HTMLDivElement>(null);
  const menuButtonRef = useRef<HTMLDivElement>(null);
  const [messageApi, contextHolder] = message.useMessage();

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        menuRef.current &&
        !menuRef.current.contains(event.target as Node) &&
        menuButtonRef.current &&
        !menuButtonRef.current.contains(event.target as Node)
      ) {
        setShowMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);
  // Initialize edit form data
  const initEditForm = () => ({
    name: role.name,
    description: role.description,
    system_prompt: role.system_prompt,
    icon: role.icon,
    is_active: role.is_active,
    enable_l0_retrieval: role.enable_l0_retrieval
  });

  const [editForm, setEditForm] = useState<UpdateRoleReq>(initEditForm());

  const handleCardClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onClick();
  };

  const handleMenuClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowMenu(!showMenu);
  };

  const handleEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowMenu(false);
    // Reset form data when opening edit modal
    setEditForm(initEditForm());
    setShowEditModal(true);
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowMenu(false);
    setShowDeleteModal(true);
  };

  const handleEditSubmit = async () => {
    await onEdit(role.id, editForm);
    setShowEditModal(false);
  };

  const handleDeleteConfirm = async () => {
    await onDelete(role.id);
    setShowDeleteModal(false);
  };

  return (
    <div className="relative border rounded-lg p-4 hover:shadow-md transition-shadow bg-white">
      {contextHolder}
      <div className="w-full text-left cursor-pointer" onClick={handleCardClick}>
        <div className="flex justify-between items-start mb-3">
          <div className="flex items-center gap-2">
            {role.icon && <img alt={role.name} className="w-6 h-6 rounded" src={role.icon} />}
            <h3 className="text-lg font-medium">{role.name}</h3>
          </div>
          <div className="flex items-center gap-2">
            <div
              ref={menuButtonRef}
              className="p-1 hover:bg-gray-100 rounded-full cursor-pointer"
              onClick={handleMenuClick}
            >
              <MoreOutlined className="text-lg text-gray-500" />
            </div>
          </div>
        </div>

        <p className="text-gray-600 text-sm line-clamp-2 mb-3">{role.description}</p>

        <div className="text-sm text-gray-500 flex justify-between">
          <span>Created {new Date(role.create_time).toLocaleDateString()}</span>
          <span>Updated {new Date(role.update_time).toLocaleDateString()}</span>
        </div>
      </div>

      {/* Dropdown menu */}
      {showMenu && (
        <div
          ref={menuRef}
          className="absolute right-4 top-12 w-32 bg-white border rounded-lg shadow-lg py-1 z-10"
        >
          <div
            className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 cursor-pointer flex items-center gap-2"
            onClick={(e) => {
              e.stopPropagation();
              setShowMenu(false);
              setShowShareModal(true);
            }}
          >
            <svg
              className="w-[14px] h-[14px]"
              fill="none"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <circle cx="18" cy="5" r="2.5" stroke="currentColor" strokeWidth="1.5" />
              <circle cx="6" cy="12" r="2.5" stroke="currentColor" strokeWidth="1.5" />
              <circle cx="18" cy="19" r="2.5" stroke="currentColor" strokeWidth="1.5" />
              <path d="M15.0355 6.41421L8.96447 10.5858" stroke="currentColor" strokeWidth="1.5" />
              <path d="M15.0355 17.5858L8.96447 13.4142" stroke="currentColor" strokeWidth="1.5" />
            </svg>
            share
          </div>
          <div
            className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 cursor-pointer flex items-center gap-2"
            onClick={handleEdit}
          >
            <svg
              className="w-[14px] h-[14px]"
              fill="none"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M16.5 3.5L20.5 7.5L7 21H3V17L16.5 3.5Z"
                stroke="currentColor"
                strokeLinejoin="round"
                strokeWidth="1.5"
              />
              <path
                d="M14 6L18 10"
                stroke="currentColor"
                strokeLinejoin="round"
                strokeWidth="1.5"
              />
            </svg>
            Edit
          </div>
          <div
            className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 text-red-600 cursor-pointer flex items-center gap-2"
            onClick={handleDelete}
          >
            <svg
              className="w-[14px] h-[14px]"
              fill="none"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path d="M4 7H20" stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" />
              <path d="M10 11V17" stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" />
              <path d="M14 11V17" stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" />
              <path
                d="M5 7L6 19C6 20.1046 6.89543 21 8 21H16C17.1046 21 18 20.1046 18 19L19 7"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="1.5"
              />
              <path
                d="M9 7V4C9 3.44772 9.44772 3 10 3H14C14.5523 3 15 3.44772 15 4V7"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="1.5"
              />
            </svg>
            Delete
          </div>
        </div>
      )}

      {/* Edit modal */}
      <Modal
        centered
        okText="Save"
        onCancel={() => {
          setShowEditModal(false);
          // Reset form data when closing modal
          setEditForm(initEditForm());
        }}
        onOk={handleEditSubmit}
        open={showEditModal}
        title="Edit Role"
      >
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
            <input
              className="w-full px-3 py-2 border rounded-lg"
              onChange={(e) => setEditForm({ ...editForm, name: e.target.value })}
              type="text"
              value={editForm.name}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
            <textarea
              className="w-full px-3 py-2 border rounded-lg"
              onChange={(e) => setEditForm({ ...editForm, description: e.target.value })}
              rows={3}
              value={editForm.description}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">System Prompt</label>
            <textarea
              className="w-full px-3 py-2 border rounded-lg"
              onChange={(e) => setEditForm({ ...editForm, system_prompt: e.target.value })}
              rows={3}
              value={editForm.system_prompt}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Icon URL</label>
            <input
              className="w-full px-3 py-2 border rounded-lg"
              onChange={(e) => setEditForm({ ...editForm, icon: e.target.value })}
              type="text"
              value={editForm.icon}
            />
          </div>
        </div>
      </Modal>

      {/* Delete confirmation modal */}
      <Modal
        centered
        okButtonProps={{ danger: true }}
        okText="Delete"
        onCancel={() => setShowDeleteModal(false)}
        onOk={handleDeleteConfirm}
        open={showDeleteModal}
        title="Delete Role"
      >
        <p>Are you sure you want to delete this role? This action cannot be undone.</p>
      </Modal>

      {/* Share modal */}
      <Modal
        footer={null}
        onCancel={() => setShowShareModal(false)}
        open={showShareModal}
        title="Share"
      >
        {isRegistered ? (
          <div className="space-y-4">
            <input
              className="w-full px-3 py-2 border rounded-lg bg-gray-50"
              readOnly
              value="https://secondme.ai/share/123456"
            />
            <button
              className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
              onClick={async () => {
                copyToClipboard('https://secondme.ai/share/123456')
                  .then(() => {
                    messageApi.success({
                      content: 'Link copied.'
                    });
                  })
                  .catch(() => {
                    messageApi.error({
                      content: 'Failed to copy, please copy manually'
                    });
                  });
              }}
            >
              Copy Link
            </button>
          </div>
        ) : (
          <div className="text-center text-red-500">Please register this Upload before sharing</div>
        )}
      </Modal>
    </div>
  );
}
