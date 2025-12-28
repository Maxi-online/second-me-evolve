import type { CommonResponse, EmptyResponse } from '../types/responseModal';
import { Request } from '../utils/request';

export interface LoadInfo {
  name: string;
  description: string;
  active?: boolean;
  email: string;
}

export interface ILoadInfo {
  id: string;
  instance_id?: string;
  name: string;
  description: string;
  status: 'online' | 'registered' | 'offline' | 'unregistered';
  created_at: string;
  updated_at: string;
  avatar_data: string | null;
  email: string;
}

export const createLoadInfo = (loadInfo: LoadInfo) => {
  return Request<CommonResponse<ILoadInfo>>({
    method: 'post',
    url: '/api/loads',
    data: loadInfo
  });
};

export const getCurrentInfo = () => {
  return Request<CommonResponse<ILoadInfo>>({
    method: 'get',
    url: '/api/loads/current'
  });
};

export const updateLoadInfo = (loadInfo: LoadInfo) => {
  return Request<CommonResponse<any>>({
    method: 'put',
    url: `/api/loads/current`,
    data: loadInfo
  });
};

export const deleteLoadInfo = (name: string) => {
  return Request<EmptyResponse>({
    method: 'delete',
    url: `/api/loads/${name}`
  });
};

export const uploadLoadAvatar = (name: string, data: { avatar_data: string }) => {
  return Request<CommonResponse<string>>({
    method: 'post',
    url: `/api/loads/${name}/avatar`,
    data: data
  });
};

export const getUploadCount = () => {
  return Request<CommonResponse<{ count: number }>>({
    method: 'get',
    url: '/api/upload/count'
  });
};
