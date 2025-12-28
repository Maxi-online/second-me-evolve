import type { CommonResponse, EmptyResponse } from '../types/responseModal';
import { Request } from '../utils/request';

export interface RegisterUploadRes {
  instance_id: string;
  status: string;
  upload_name: string;
  ws_url: string;
}

export interface Upload {
  instance_id?: string;
  upload_name: string;
  description?: string;
  email?: string;
}

export interface IUpdateUploadRes {
  instance_id: string;
  status: string;
  upload_name: string;
  updated_fields: string[];
}

export interface IUploadInfo {
  description: string;
  email: string;
  instance_id: string;
  status: string;
  upload_name: string;
}

interface IUploadList {
  items: IUploadInfo[];
  pagination: {
    page_no: number;
    page_size: number;
    total: number;
    total_pages: number;
  };
}

export const registerUpload = (data: Upload) => {
  return Request<CommonResponse<IUploadInfo>>({
    method: 'POST',
    url: '/api/upload/register',
    data
  });
};

export const getUploadStatus = () => {
  return Request<CommonResponse<IUploadInfo>>({
    method: 'GET',
    url: `/api/upload/status`
  });
};

export const deleteUpload = () => {
  return Request<CommonResponse<Upload>>({
    method: 'DELETE',
    url: `/api/upload`
  });
};

export const getUploadList = (data?: { page_no?: number; page_size?: number }) => {
  return Request<CommonResponse<IUploadList>>({
    method: 'GET',
    url: '/api/upload',
    params: data
  });
};

export const connectUpload = () => {
  return Request<EmptyResponse>({
    method: 'POST',
    url: `/api/upload/connect`
  });
};

export const updateUpload = (data: Upload) => {
  return Request<CommonResponse<IUpdateUploadRes>>({
    method: 'PUT',
    url: `/api/upload`,
    data: {
      upload_name: data.upload_name,
      description: data.description,
      email: data.email
    }
  });
};
