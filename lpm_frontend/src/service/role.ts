import { Request } from '../utils/request';
import type { CommonResponse, EmptyResponse } from '../types/responseModal';

export interface CreateRoleReq {
  name: string;
  description: string;
  system_prompt: string;
  icon: string;
  enable_l0_retrieval: boolean;
  enable_l1_retrieval?: boolean;
}

export interface RoleRes {
  id: number;
  uuid: string;
  name: string;
  description: string;
  system_prompt: string;
  icon: string;
  is_active: boolean;
  create_time: string;
  update_time: string;
  enable_l0_retrieval: boolean;
  enable_l1_retrieval?: boolean;
}

export interface uploadRoleReq {
  role_id: string;
  name: string;
  description?: string;
  system_prompt: string;
  icon?: string;
  is_active?: boolean;
  enable_l0_retrieval?: boolean;
  enable_l1_retrieval?: boolean;
}

export interface UpdateRoleReq extends CreateRoleReq {
  is_active?: boolean;
}

export const uploadRole = (data: { role_id: string }) => {
  return Request<CommonResponse<RoleRes>>({
    method: 'POST',
    url: '/api/kernel2/roles/share',
    data
  });
};

export const createRole = (data: CreateRoleReq) => {
  return Request<CommonResponse<RoleRes>>({
    method: 'POST',
    url: '/api/kernel2/roles',
    data
  });
};

export const getRoleList = () => {
  return Request<CommonResponse<RoleRes[]>>({
    method: 'GET',
    url: '/api/kernel2/roles'
  });
};

export const getRole = (uuid: string) => {
  return Request<CommonResponse<RoleRes>>({
    method: 'GET',
    url: `/api/kernel2/roles/${uuid}`
  });
};

export const updateRole = (uuid: string, data: UpdateRoleReq) => {
  return Request<CommonResponse<RoleRes>>({
    method: 'PUT',
    url: `/api/kernel2/roles/${uuid}`,
    data
  });
};

export const deleteRole = (uuid: string) => {
  return Request<CommonResponse<EmptyResponse>>({
    method: 'DELETE',
    url: `/api/kernel2/roles/${uuid}`
  });
};
