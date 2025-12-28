'use client';

import { useState, useEffect, useMemo } from 'react';
import CreateRoleModal from '@/components/roleplay/CreateRoleModal';
import EditRoleModal from '@/components/roleplay/EditRoleModal';
import DeleteRoleModal from '@/components/roleplay/DeleteRoleModal';
import ShareRoleModal from '@/components/roleplay/ShareRoleModal';
import {
  createRole,
  getRoleList,
  updateRole,
  deleteRole,
  uploadRole,
  type RoleRes,
  type CreateRoleReq,
  type UpdateRoleReq
} from '@/service/role';
import { message } from 'antd';
import { useLoadInfoStore } from '@/store/useLoadInfoStore';
import { EVENT } from '@/utils/event';

export default function Roleplay() {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showShareModal, setShowShareModal] = useState(false);
  const [roles, setRoles] = useState<RoleRes[]>([]);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [editing, setEditing] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [selectedRole, setSelectedRole] = useState<RoleRes | null>(null);
  const [editForm, setEditForm] = useState<UpdateRoleReq>({
    name: '',
    description: '',
    system_prompt: '',
    icon: '',
    is_active: true,
    enable_l0_retrieval: true
  });

  const loadInfo = useLoadInfoStore((state) => state.loadInfo);
  const isRegistered = useMemo(() => {
    return loadInfo?.status === 'online';
  }, [loadInfo]);

  const [messageApi, contextHolder] = message.useMessage();

  useEffect(() => {
    if (!isRegistered) {
      dispatchEvent(new Event(EVENT.SHOW_REGISTER_MODAL));
    }
  }, [isRegistered]);

  // Load role list
  const loadRoles = async () => {
    setLoading(true);

    try {
      const res = await getRoleList();

      setRoles(res.data.data);
    } catch (error) {
      messageApi.error(`Failed to load roles: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // Initial loading
  useEffect(() => {
    loadRoles();
  }, []);

  const handleCreateRole = async (roleData: CreateRoleReq) => {
    setCreating(true);

    createRole(roleData)
      .then((res) => {
        if (res.data.code !== 0) {
          throw new Error(res.data.message);
        }

        setRoles((prev) => [res.data.data, ...prev]);
        setShowCreateModal(false);
        messageApi.success('Role created successfully');
      })
      .catch((error: any) => {
        messageApi.error(
          `Failed to create role: ${error?.response?.data?.message || error.message}`
        );
      })
      .finally(() => {
        setCreating(false);
      });
  };

  const handleShareRole = (role: RoleRes) => {
    if (!isRegistered) {
      messageApi.error('Please join AI network first');

      return;
    }

    const data = {
      role_id: role.uuid
    };

    uploadRole(data).then((res) => {
      if (res.data.code === 0) {
        setShowShareModal(true);
      } else {
        messageApi.error('Failed to share role');
      }
    });
  };

  const handleRoleClick = (role_id: string) => {
    // Open in a new tab
    window.open(`/standalone/role/${role_id}`, '_blank');
  };

  const handleEditRole = async (uuid: string, data: UpdateRoleReq) => {
    console.log(uuid, 'uuid');

    setEditing(true);

    try {
      const res = await updateRole(uuid, data);

      if (res.data.code === 0) {
        setRoles((prev) => prev.map((role) => (role.uuid === uuid ? res.data.data : role)));
        setShowEditModal(false);
        messageApi.success('Role updated successfully');
      } else {
        messageApi.error('Failed to update role');
      }
    } catch (error) {
      messageApi.error(`Failed to update role: ${error}`);
    } finally {
      setEditing(false);
      setSelectedRole(null);
    }
  };

  const handleDeleteRole = async (uuid: string) => {
    setDeleting(true);

    try {
      const res = await deleteRole(uuid);

      if (res.data.code === 0) {
        setRoles((prev) => prev.filter((role) => role.uuid !== uuid));
        setShowDeleteModal(false);
        messageApi.success('Role deleted successfully');
      } else {
        messageApi.error('Failed to delete role');
      }
    } catch (error) {
      messageApi.error(`Failed to delete role: ${error}`);
    } finally {
      setDeleting(false);
      setSelectedRole(null);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {contextHolder}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">My Roleplay Apps</h1>
          <p className="mt-2 text-sm text-gray-600">
            {`Create and manage specialized 'roles' for your Second Me. Each role has a specific
            purpose, style, or domain.`}
          </p>
        </div>
        <div className="flex gap-3">
          <button
            className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all hover:-translate-y-0.5 disabled:bg-blue-400 disabled:hover:translate-y-0"
            disabled={loading || creating}
            onClick={() => setShowCreateModal(true)}
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                d="M12 4v16m8-8H4"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
              />
            </svg>
            Create Roleplay App
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center items-center py-12">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600" />
        </div>
      ) : roles.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 px-4">
          <div className="text-center mb-6">
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
              />
            </svg>
            <h3 className="mt-4 text-lg font-medium text-gray-900">No apps created</h3>
            <p className="mt-1 text-sm text-gray-500">
              Get started by creating your first SecondMe app!
            </p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {roles.map((role) => (
            <div
              key={role.uuid}
              className="group relative bg-white rounded-lg shadow-sm overflow-hidden ring-1 ring-black/5 hover:shadow-lg transition-all duration-200 hover:-translate-y-1 cursor-pointer"
              onClick={() => handleRoleClick(role.uuid)}
            >
              <div className="p-6">
                <div className="flex flex-col mb-4">
                  <div className="flex items-start">
                    <h3 className="text-lg font-semibold line-clamp-1 text-gray-900 group-hover:text-blue-600 transition-colors">
                      {role.name}
                    </h3>
                    <div className="ml-auto flex space-x-2">
                      <button
                        className="text-gray-400 hover:text-blue-500 transition-colors"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedRole(role);
                          setEditForm({
                            name: role.name,
                            description: role.description,
                            system_prompt: role.system_prompt,
                            icon: role.icon,
                            is_active: role.is_active,
                            enable_l0_retrieval: role.enable_l0_retrieval
                          });
                          setShowEditModal(true);
                        }}
                      >
                        <svg
                          className="w-5 h-5"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            d="M16.5 3.5L20.5 7.5L7 21H3V17L16.5 3.5Z"
                            strokeLinejoin="round"
                            strokeWidth="1.5"
                          />
                          <path d="M14 6L18 10" strokeLinejoin="round" strokeWidth="1.5" />
                        </svg>
                      </button>
                      <button
                        className={`text-gray-400 hover:text-blue-500 transition-colors`}
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedRole(role);
                          handleShareRole(role);
                        }}
                      >
                        <svg
                          className="w-5 h-5"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
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
                          setSelectedRole(role);
                          setShowDeleteModal(true);
                        }}
                      >
                        <svg
                          className="w-5 h-5"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                          />
                        </svg>
                      </button>
                    </div>
                  </div>
                  <span className="mt-1 w-fit inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                    Roleplay App
                  </span>
                </div>
                <p className="text-sm text-gray-600 line-clamp-2 mb-4">{role.description}</p>
                <div className="flex items-center justify-end text-sm text-gray-500">
                  <button className="inline-flex items-center text-gray-600 hover:text-blue-600 transition-colors">
                    View App
                    <svg
                      className="ml-2 w-4 h-4 transition-transform group-hover:translate-x-1"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        d="M9 5l7 7-7 7"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                      />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <CreateRoleModal
        loading={creating}
        onClose={() => setShowCreateModal(false)}
        onSubmit={handleCreateRole}
        open={showCreateModal}
      />

      <EditRoleModal
        initialData={editForm}
        loading={editing}
        onClose={() => {
          setShowEditModal(false);
          setSelectedRole(null);
        }}
        onSubmit={(data) => handleEditRole(selectedRole!.uuid, data)}
        open={showEditModal}
      />

      <DeleteRoleModal
        loading={deleting}
        onClose={() => {
          setShowDeleteModal(false);
          setSelectedRole(null);
        }}
        onConfirm={() => handleDeleteRole(selectedRole!.uuid)}
        open={showDeleteModal}
      />

      <ShareRoleModal
        isRegistered={isRegistered}
        onClose={() => {
          setShowShareModal(false);
          setSelectedRole(null);
        }}
        open={showShareModal}
        uuid={selectedRole?.uuid || ''}
      />

      {/* Example section */}
      <div className="relative z-10 mt-8 text-right text-sm text-gray-500">
        <p className="text-right mb-2">Try example:</p>
        <a
          className="hover:text-gray-700 hover:underline"
          href="https://app.secondme.io/example/ama"
          rel="noopener noreferrer"
          target="_blank"
        >
          Felix AMA (Roleplay)
        </a>
      </div>
    </div>
  );
}
