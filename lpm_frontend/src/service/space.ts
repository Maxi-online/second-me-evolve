import { Request } from '../utils/request';
import type { CommonResponse, EmptyResponse } from '../types/responseModal';

export interface SpaceMessage {
  id: string;
  content: string;
  create_time: string;
  message_type: string;
  role: string;
  round: number;
  sender_endpoint: string;
  space_id: string;
}

export interface ParticipantInfo {
  url: string;
  role_description?: string;
}

export interface SpaceInfo {
  conclusion: string | null;
  create_time: string;
  host: string;
  id: string;
  objective: string;
  participants: string[];
  participants_info?: ParticipantInfo[];
  title: string;
  status?: number; // 1-Initialized 2-In Discussion 3-Discussion Interrupted 4-Discussion Ended
  messages?: SpaceMessage[];
}

interface CreateSpaceReq {
  title: string;
  objective: string;
  host: string;
  participants: string[];
}

interface IShareSpace {
  space: SpaceInfo;
  space_share_id: string;
}

export const createSpace = (data: CreateSpaceReq) => {
  return Request<CommonResponse<SpaceInfo>>({
    method: 'POST',
    url: '/api/space/create',
    data
  });
};

export const getSpaceDetail = (space_id: string) => {
  return Request<CommonResponse<SpaceInfo>>({
    method: 'GET',
    url: `/api/space/${space_id}`
  });
};

export const getAllSpaces = () => {
  return Request<CommonResponse<SpaceInfo[]>>({
    method: 'GET',
    url: '/api/space/all'
  });
};

export const startSpace = (space_id: string) => {
  return Request<CommonResponse<SpaceInfo>>({
    method: 'POST',
    url: `/api/space/${space_id}/start`
  });
};

export const deleteSpace = (space_id: string) => {
  return Request<EmptyResponse>({
    method: 'DELETE',
    url: `/api/space/${space_id}`
  });
};

export const shareSpace = (space_id: string) => {
  return Request<CommonResponse<IShareSpace>>({
    method: 'POST',
    url: `/api/space/${space_id}/share`
  });
};
