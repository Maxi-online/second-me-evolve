import { Request } from '../utils/request';
import type { CommonResponse } from '../types/responseModal';

interface Bio {
  content: string;
  content_third_view: string;
  shades: any[];
  summary: string;
  summary_third_view: string;
}

interface ChunkTopic {
  chunk_id: string;
  tags: string[];
  topic: string;
}

interface Cluster {
  cluster_center: any;
  cluster_id: any;
  memory_ids: string[];
}

export interface GlobalBioResponse {
  bio: Bio;
  chunk_topics: ChunkTopic[];
  clusters: Cluster[];
  version: number;
}

export interface StatusBioResponse {
  content: string;
  content_third_view: string;
  create_time: string;
  summary: string;
  summary_third_view: string;
  update_time: string;
}

export interface BioVersion {
  create_time: string;
  description: string;
  status: string;
  version: number;
}

export const getGlobalBioVersion = () => {
  return Request<CommonResponse<BioVersion[]>>({
    method: 'get',
    url: '/api/kernel/l1/global/versions'
  });
};

export const getGlobalBio = (version: number) => {
  return Request<CommonResponse<GlobalBioResponse>>({
    method: 'get',
    url: `/api/kernel/l1/global/version/${version}`
  });
};

export const getStatusBio = () => {
  return Request<CommonResponse<StatusBioResponse>>({
    method: 'get',
    url: '/api/kernel/l1/status_bio/get'
  });
};
