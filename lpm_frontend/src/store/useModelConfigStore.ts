import { create } from 'zustand';
import {
  getModelConfig,
  type IBaseModelParams,
  type IModelConfig,
  type IThinkingModelParams
} from '@/service/modelConfig';

interface ModelConfigState {
  modelConfig: IModelConfig;
  baseModelConfig: IBaseModelParams;
  thinkingModelConfig: IThinkingModelParams;
  fetchModelConfig: () => Promise<void>;
  updateModelConfig: (config: IModelConfig) => void;
  deleteModelConfig: () => void;
  updateBaseModelConfig: (config: IBaseModelParams) => void;
  updateThinkingModelConfig: (config: IThinkingModelParams) => void;
}

export const useModelConfigStore = create<ModelConfigState>((set, get) => ({
  modelConfig: {} as IModelConfig,
  baseModelConfig: {} as IBaseModelParams,
  thinkingModelConfig: {} as IThinkingModelParams,
  fetchModelConfig: async () => {
    return getModelConfig()
      .then((res) => {
        if (res.data.code !== 0) {
          throw new Error(res.data.message);
        }

        const { thinking_model_name, thinking_api_key, thinking_endpoint, ...baseModelConfig } =
          res.data.data;

        set({
          modelConfig: { ...(get().modelConfig as IModelConfig), ...res.data.data },
          baseModelConfig: { ...(get().baseModelConfig as IBaseModelParams), ...baseModelConfig },
          thinkingModelConfig: {
            ...(get().thinkingModelConfig as IThinkingModelParams),
            thinking_model_name,
            thinking_api_key,
            thinking_endpoint
          }
        });
      })
      .catch((error) => {
        console.error(error.message || 'Failed to fetch model config');
      });
  },
  updateModelConfig(config: IModelConfig) {
    const { thinking_model_name, thinking_api_key, thinking_endpoint, ...baseModelConfig } = config;

    set({
      modelConfig: { ...(get().modelConfig as IModelConfig), ...config },
      baseModelConfig: { ...(get().baseModelConfig as IBaseModelParams), ...baseModelConfig },
      thinkingModelConfig: {
        ...(get().thinkingModelConfig as IThinkingModelParams),
        thinking_model_name,
        thinking_api_key,
        thinking_endpoint
      }
    });
  },
  deleteModelConfig() {
    set({
      modelConfig: {} as IModelConfig,
      baseModelConfig: {} as IBaseModelParams,
      thinkingModelConfig: {} as IThinkingModelParams
    });
  },
  updateBaseModelConfig(config: IBaseModelParams) {
    set({
      baseModelConfig: { ...(get().baseModelConfig as IBaseModelParams), ...config }
    });

    set({
      modelConfig: {
        ...(get().modelConfig as IModelConfig),
        ...(get().baseModelConfig as IBaseModelParams),
        ...config
      }
    });
  },
  updateThinkingModelConfig(config: IThinkingModelParams) {
    set({
      thinkingModelConfig: { ...(get().thinkingModelConfig as IThinkingModelParams), ...config }
    });

    set({
      modelConfig: {
        ...(get().modelConfig as IModelConfig),
        ...(get().thinkingModelConfig as IThinkingModelParams),
        ...config
      }
    });
  }
}));
