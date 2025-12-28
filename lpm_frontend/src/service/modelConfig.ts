import { Request } from '../utils/request';
import type { CommonResponse, EmptyResponse } from '../types/responseModal';

export type IModelConfig = IBaseModelParams & IThinkingModelParams;

export interface IBaseModelParams {
  id: number;
  provider_type: string;
  key: string;
  chat_endpoint: string;
  chat_api_key: string;
  chat_model_name: string;
  embedding_endpoint: string;
  embedding_api_key: string;
  embedding_model_name: string;
  created_at: string;
  updated_at: string;
}

export interface IThinkingModelParams {
  thinking_model_name: string;
  thinking_api_key: string;
  thinking_endpoint: string;
}

export const getModelConfig = () => {
  return Request<CommonResponse<IModelConfig>>({
    method: 'get',
    url: `/api/user-llm-configs`
  });
};

export const updateModelConfig = (data: IModelConfig) => {
  return Request<CommonResponse<IModelConfig>>({
    method: 'put',
    url: `/api/user-llm-configs`,
    data
  });
};

export const updateThinkingConfig = (data: IThinkingModelParams) => {
  return Request<CommonResponse<IThinkingModelParams>>({
    method: 'put',
    url: `/api/user-llm-configs/thinking`,
    data
  });
};

export const deleteModelConfig = () => {
  return Request<EmptyResponse>({
    method: 'delete',
    url: `/api/user-llm-configs/key`
  });
};
