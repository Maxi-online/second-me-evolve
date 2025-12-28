import { updateModelConfig } from '../../service/modelConfig';
import { useModelConfigStore } from '../../store/useModelConfigStore';
import { Input, Modal, Radio } from 'antd';
import Image from 'next/image';
import { useCallback, useEffect, useState } from 'react';
import { QuestionCircleOutlined } from '@ant-design/icons';

interface IProps {
  open: boolean;
  onClose: () => void;
}

const options = [
  {
    label: 'None',
    value: ''
  },
  {
    label: 'OpenAI',
    value: 'openai'
  },
  {
    label: 'Custom',
    value: 'litellm'
  }
];

const ModelConfigModal = (props: IProps) => {
  const { open, onClose } = props;
  const modelConfig = useModelConfigStore((store) => store.modelConfig);
  const baseModelConfig = useModelConfigStore((store) => store.baseModelConfig);
  const updateBaseModelConfig = useModelConfigStore((store) => store.updateBaseModelConfig);
  const fetchModelConfig = useModelConfigStore((store) => store.fetchModelConfig);
  const localProviderType = useModelConfigStore((store) => store.modelConfig.provider_type);
  const [modelType, setModelType] = useState<string>('');

  useEffect(() => {
    if (open) {
      fetchModelConfig();
    }
  }, [open]);

  useEffect(() => {
    setModelType(localProviderType);
  }, [localProviderType]);

  const renderEmpty = () => {
    return (
      <div className="flex flex-col items-center">
        <Image
          alt="SecondMe Logo"
          className="object-contain"
          height={40}
          src="/images/single_logo.png"
          width={120}
        />
        <div className="text-gray-500 text-[18px] leading-[32px]">
          Please Choose OpenAI or Custom
        </div>
      </div>
    );
  };

  const renderOpenai = useCallback(() => {
    return (
      <div className="flex flex-col w-full gap-4">
        <div className="p-4 border rounded-lg hover:shadow-md transition-shadow">
          <label className="block text-sm font-medium text-gray-700 mb-1">API Key</label>
          <Input.Password
            onChange={(e) => {
              updateBaseModelConfig({ ...baseModelConfig, key: e.target.value });
            }}
            placeholder="Enter your OpenAI API key"
            value={baseModelConfig.key}
          />
          <div className="mt-2 text-sm text-gray-500">
            You can get your API key from{' '}
            <a
              className="text-blue-500 hover:underline"
              href="https://platform.openai.com/settings/organization/api-keys"
              rel="noopener noreferrer"
              target="_blank"
            >
              OpenAI API Keys page
            </a>
            .
          </div>
        </div>
      </div>
    );
  }, [baseModelConfig]);

  const renderCustom = useCallback(() => {
    return (
      <div className="flex flex-col w-full gap-6 p-4">
        <div className="p-4 border rounded-lg hover:shadow-md transition-shadow">
          <label className="block text-sm font-medium text-gray-700 mb-1">Chat</label>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex flex-col">
              <div className="text-sm font-medium text-gray-700 mb-1">Model Name</div>
              <Input
                autoCapitalize="off"
                autoComplete="off"
                autoCorrect="off"
                className="w-full"
                data-form-type="other"
                onChange={(e) => {
                  updateBaseModelConfig({ ...baseModelConfig, chat_model_name: e.target.value });
                }}
                spellCheck="false"
                value={baseModelConfig.chat_model_name}
              />
            </div>

            <div className="flex flex-col">
              <div className="text-sm font-medium text-gray-700 mb-1">API Key</div>
              <Input.Password
                autoCapitalize="off"
                autoComplete="new-password"
                autoCorrect="off"
                className="w-full"
                data-form-type="other"
                onChange={(e) => {
                  updateBaseModelConfig({ ...baseModelConfig, chat_api_key: e.target.value });
                }}
                spellCheck="false"
                value={baseModelConfig.chat_api_key}
              />
            </div>
          </div>

          <div className="mt-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">API Endpoint</label>
            <Input
              autoComplete="off"
              className="w-full"
              onChange={(e) => {
                updateBaseModelConfig({ ...baseModelConfig, chat_endpoint: e.target.value });
              }}
              value={baseModelConfig.chat_endpoint}
            />
          </div>
        </div>

        <div className="p-4 border rounded-lg hover:shadow-md transition-shadow">
          <label className="block text-sm font-medium text-gray-700 mb-1">Embedding</label>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Model Name</label>
              <Input
                className="w-full"
                onChange={(e) => {
                  updateBaseModelConfig({
                    ...baseModelConfig,
                    embedding_model_name: e.target.value
                  });
                }}
                value={baseModelConfig.embedding_model_name}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">API Key</label>
              <Input.Password
                className="w-full"
                onChange={(e) => {
                  updateBaseModelConfig({ ...baseModelConfig, embedding_api_key: e.target.value });
                }}
                value={baseModelConfig.embedding_api_key}
              />
            </div>
          </div>

          <div className="mt-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">API Endpoint</label>
            <Input
              className="w-full"
              onChange={(e) => {
                updateBaseModelConfig({ ...baseModelConfig, embedding_endpoint: e.target.value });
              }}
              value={baseModelConfig.embedding_endpoint}
            />
          </div>
        </div>
      </div>
    );
  }, [baseModelConfig, updateBaseModelConfig]);

  const handleUpdate = () => {
    updateModelConfig(modelConfig)
      .then((res) => {
        if (res.data.code == 0) {
          onClose();
        } else {
          throw new Error(res.data.message);
        }
      })
      .catch((error) => {
        console.error(error.message || 'Failed to update model config');
      });
  };

  const renderMainContent = useCallback(() => {
    if (!modelType) {
      return renderEmpty();
    }

    if (modelType === 'openai') {
      return renderOpenai();
    }

    return renderCustom();
  }, [modelType, renderOpenai, renderCustom]);

  return (
    <Modal
      centered
      destroyOnClose
      okButtonProps={{ disabled: !modelType }}
      onCancel={onClose}
      onOk={handleUpdate}
      open={open}
      title={
        <div className="flex items-center gap-2">
          <div className="text-xl font-semibold leading-6 text-gray-900">
            Support Model Configuration
          </div>
          <a
            className="text-gray-500 hover:text-gray-700"
            href="https://secondme.gitbook.io/secondme/guides/create-second-me/support-model-config"
            rel="noreferrer"
            target="_blank"
          >
            <QuestionCircleOutlined />
          </a>
        </div>
      }
    >
      <div className="flex flex-col items-center">
        <div className="flex flex-col items-center gap-2">
          <p className="mb-1 text-sm text-gray-500">
            Configure models used for training data synthesis for Second Me, and as external
            reference models that Second Me can consult during usage.
          </p>
          <Radio.Group
            buttonStyle="solid"
            onChange={(e) => {
              setModelType(e.target.value);
              updateBaseModelConfig({ ...baseModelConfig, provider_type: e.target.value });
            }}
            optionType="button"
            options={options}
            value={modelType ? modelType : ''}
          />
        </div>
        <div className="w-full border-t border-gray-200 mt-1 mb-2" />
        {renderMainContent()}
        <div className="w-full border-t border-gray-200 mt-4" />
      </div>
    </Modal>
  );
};

export default ModelConfigModal;
