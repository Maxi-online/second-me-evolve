'use client';

import { useState } from 'react';
import type { UploadProps } from 'antd';
import { Upload, Input, Button, message } from 'antd';
import {
  CheckCircleFilled,
  InboxOutlined,
  FileOutlined,
  FolderOutlined,
  DesktopOutlined
} from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';
import { uploadMemory } from '@/service/memory';
import { EVENT } from '@/utils/event';

interface UploadMemoriesProps {
  onFileUpload: (files: any[]) => void;
}

const { TextArea } = Input;

// Use regular components instead of styled-components
const GlobalStyle: React.FC<{ children?: React.ReactNode }> = ({
  children
}: {
  children?: React.ReactNode;
}) => (
  <div className="[&_.custom-message-success_.ant-message-notice-content]:bg-[#f6ffed] [&_.custom-message-success_.ant-message-notice-content]:border [&_.custom-message-success_.ant-message-notice-content]:border-[#b7eb8f] [&_.custom-message-success_.ant-message-notice-content]:rounded [&_.custom-message-success_.ant-message-notice-content]:p-2 [&_.custom-message-success_.ant-message-notice-content]:shadow-md">
    {children}
  </div>
);

const UploadTypeContainer: React.FC<{ children: React.ReactNode }> = ({
  children
}: {
  children: React.ReactNode;
}) => <div className="flex w-full gap-2 mb-4">{children}</div>;

interface UploadTypeBoxProps {
  active?: boolean;
  disabled?: boolean;
  children: React.ReactNode;
  onClick?: () => void;
}

const UploadTypeBox: React.FC<UploadTypeBoxProps> = ({
  active,
  disabled,
  children,
  onClick
}: UploadTypeBoxProps) => (
  <div
    className={`
      flex-1 p-4 rounded-lg cursor-pointer flex flex-col items-center gap-2 transition-all duration-300
      ${disabled ? 'opacity-50 pointer-events-none' : 'opacity-100'}
      ${
        active
          ? 'bg-gradient-to-b from-[#F8FAFF] to-[#EEF3FF] border border-[#4080FF] shadow-[0_2px_8px_rgba(64,128,255,0.15)]'
          : 'bg-white border border-[#E5E6EB] hover:border-[#4080FF] hover:bg-[#F8F9FC] hover:shadow-[0_2px_6px_rgba(0,0,0,0.05)]'
      }
      [&_.icon]:text-2xl [&_.icon]:${active ? 'text-[#4080FF]' : 'text-[#86909C]'}
      [&_.text]:text-sm [&_.text]:font-medium [&_.text]:${active ? 'text-[#4080FF]' : 'text-[#4E5969]'}
    `}
    onClick={onClick}
  >
    {children}
  </div>
);

interface TabContentProps {
  isTextArea?: boolean;
  children: React.ReactNode;
}

const TabContent: React.FC<TabContentProps> = ({ isTextArea, children }: TabContentProps) => (
  <div
    className={`
    bg-white rounded-lg p-4 shadow-[0_4px_16px_rgba(0,0,0,0.08)]
    ${isTextArea ? 'min-h-[240px] items-start' : 'min-h-[200px] items-center'}
    flex justify-center border border-[#F0F0F0]
  `}
  >
    {children}
  </div>
);

const UploadArea: React.FC<{ children: React.ReactNode }> = ({
  children
}: {
  children: React.ReactNode;
}) => (
  <div
    className="
    border-2 border-dashed border-[#E5E6EB] rounded-lg bg-[#FAFBFC] p-6 text-center cursor-pointer transition-all duration-300 w-full
    hover:border-[#4080FF] hover:bg-[#F5F8FF] hover:shadow-[0_0_0_4px_rgba(64,128,255,0.08)]
    [&_.upload-icon]:text-2xl [&_.upload-icon]:text-[#4080FF] [&_.upload-icon]:mb-2
    [&_.upload-text]:text-[#4E5969] [&_.upload-text]:mb-1 [&_.upload-text]:font-medium
    [&_.browse-link]:text-[#4080FF] [&_.browse-link]:font-medium [&_.browse-link]:no-underline hover:[&_.browse-link]:underline
    [&_.file-types]:text-[#86909C] [&_.file-types]:text-xs [&_.file-types]:mt-2
  "
  >
    {children}
  </div>
);

const TextContainer: React.FC<{ children: React.ReactNode }> = ({
  children
}: {
  children: React.ReactNode;
}) => <div className="flex flex-col gap-3 w-full h-full">{children}</div>;

const SaveButton: React.FC<React.ComponentProps<typeof Button>> = (props) => (
  <Button
    {...props}
    className="self-start px-4 h-8 flex items-center justify-center rounded-lg shadow-sm hover:shadow-md transition-all duration-300 font-medium bg-gradient-to-r from-[#4080FF] to-[#3A75E6] hover:from-[#3A75E6] hover:to-[#3369D3]"
  />
);

export default function UploadMemories({ onFileUpload }: UploadMemoriesProps) {
  const [text, setText] = useState('');
  const [activeTab, setActiveTab] = useState('text');
  const [uploadedFiles] = useState(new Set<string>());

  const showSuccessMessage = () => {
    message.success({
      content: 'Successfully added text content',
      icon: <CheckCircleFilled style={{ color: '#52c41a' }} />,
      className: 'custom-message-success'
    });
  };

  const handleFileUpload = async (file: File | UploadFile) => {
    // Check file type
    const extension = (file instanceof File ? file.name : file.name)
      .split('.')
      .pop()
      ?.toLowerCase();

    if (!extension || !['pdf', 'txt', 'md'].includes(extension)) {
      message.info('Only .pdf, .txt and .md files are supported');

      return false;
    }

    // Get filename without path
    const fullName = file instanceof File ? file.name : file.name;
    const fileName = fullName.split('/').pop() || fullName;
    const fileSize = file instanceof File ? file.size : file.size;
    const fileKey = `${fileName}_${fileSize}`;

    try {
      const formData = new FormData();

      if (file instanceof File) {
        // Create a new File object, using only the filename
        const newFile = new File([file], fileName, { type: file.type });

        formData.append('file', newFile);
      } else {
        // Ensure we get the original File object from UploadFile
        const originalFile = file.originFileObj;

        if (!originalFile) {
          throw new Error('Cannot get file content');
        }

        // Create a new File object, using only the filename
        const newFile = new File([originalFile], fileName, {
          type: originalFile.type
        });

        formData.append('file', newFile);
      }

      const res = await uploadMemory(formData);

      if (res.data.code !== 0) {
        throw new Error(res.data.message);
      }

      uploadedFiles.add(fileKey);
      onFileUpload([res.data.data]);
      showSuccessMessage();

      return true;
    } catch (error: any) {
      message.error(`${fileName} upload failed: ${error.message}`);

      return false;
    }
  };

  const handleTextSubmit = async () => {
    if (text.trim()) {
      const formData = new FormData();
      const blob = new Blob([text], { type: 'text/plain' });
      const file = new File([blob], `note_${new Date().getTime()}.txt`, {
        type: 'text/plain'
      });

      formData.append('file', file);

      try {
        const res = await uploadMemory(formData);

        if (res.data.code !== 0) {
          throw new Error(res.data.message);
        }

        dispatchEvent(new Event(EVENT.REFRESH_MEMORIES));
        onFileUpload([res.data.data]);
        showSuccessMessage();
        setText('');
      } catch (error: any) {
        message.error(`Upload failed: ${error.message}`);
      }
    }
  };

  // Single file upload configuration
  const fileProps: UploadProps = {
    name: 'file',
    multiple: false,
    showUploadList: false,
    customRequest: async ({ file, onSuccess, onError }) => {
      try {
        const fileToUpload = file instanceof File ? file : (file as any).originFileObj;

        if (!fileToUpload) {
          message.error('Unable to get file content');

          return;
        }

        // Check file type
        const extension = fileToUpload.name.split('.').pop()?.toLowerCase();

        if (!extension || !['pdf', 'txt', 'md'].includes(extension)) {
          message.info('Only .pdf, .txt and .md files are supported');

          return;
        }

        const success = await handleFileUpload(fileToUpload);

        if (success) {
          onSuccess?.(file);
          dispatchEvent(new Event(EVENT.REFRESH_MEMORIES));
        } else {
          onError?.(new Error('Upload failed'));
        }
      } catch (error: any) {
        message.error(error.message || 'Upload failed');
        onError?.(new Error('Upload failed'));
      }
    },
    onChange: (info) => {
      if (info.file.status === 'done') {
        // showSuccessMessage();
      } else if (info.file.status === 'error') {
        message.error(`${info.file.name} upload failed`);
      }
    }
  };

  // Configuration for folder upload
  const folderProps: UploadProps = {
    showUploadList: false,
    directory: true,
    customRequest: async ({ file, onSuccess }) => {
      try {
        const fileToUpload = file instanceof File ? file : (file as any).originFileObj;

        if (!fileToUpload) {
          message.error('Unable to get file content');

          return;
        }

        // Check file type
        const extension = fileToUpload.name.split('.').pop()?.toLowerCase();

        if (!extension || !['pdf', 'txt', 'md'].includes(extension)) {
          // Skip unsupported file types without error
          onSuccess?.(file);

          return;
        }

        const success = await handleFileUpload(fileToUpload);

        if (success) {
          onSuccess?.(file);
          dispatchEvent(new Event(EVENT.REFRESH_MEMORIES));
        }
      } catch (error: any) {
        message.error(error.message || 'Upload failed');
      }
    },
    onChange: (_info) => {
      // Display total number of files
      // if (info.fileList.length > 0) {
      //   const validFiles = info.fileList.filter((file) => {
      //     const extension = file.name.split('.').pop()?.toLowerCase();
      //     return extension === 'pdf' || extension === 'txt' || extension === 'md';
      //   });
      // }
    }
  };

  const uploadTypes = [
    { key: 'text', icon: <FileOutlined className="icon" />, text: 'Text', disabled: false },
    { key: 'file', icon: <InboxOutlined className="icon" />, text: 'File', disabled: false },
    { key: 'folder', icon: <FolderOutlined className="icon" />, text: 'Folder', disabled: false },
    {
      key: 'software',
      icon: <DesktopOutlined className="icon" />,
      text: 'Software Integration',
      disabled: true
    },
    {
      key: 'wearable',
      icon: (
        <svg
          aria-hidden="true"
          className="icon"
          fill="currentColor"
          height="1em"
          viewBox="0 0 24 24"
          width="1em"
        >
          <path d="M6 10c-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4-1.79-4-4-4zm0 6c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm12-6c-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4-1.79-4-4-4zm0 6c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm-6-5.99h-2v2h2v-2z" />
        </svg>
      ),
      text: 'Wearable Integration',
      disabled: true
    }
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'text':
        return (
          <TabContent isTextArea>
            <TextContainer>
              <TextArea
                className="hover:border-[#4080FF] focus:border-[#4080FF] focus:shadow-[0_0_0_2px_rgba(64,128,255,0.2),inset_0_2px_4px_rgba(0,0,0,0.03)] flex-1"
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter your text here..."
                style={{
                  resize: 'none',
                  minHeight: '180px',
                  padding: '16px',
                  fontSize: '14px',
                  lineHeight: '1.6',
                  border: '1px solid #E5E6EB',
                  borderRadius: '12px',
                  backgroundColor: '#FAFAFA',
                  boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.03)',
                  transition: 'all 0.3s ease'
                }}
                value={text}
              />
              <SaveButton onClick={handleTextSubmit} size="large" type="primary">
                Save Text
              </SaveButton>
            </TextContainer>
          </TabContent>
        );
      case 'file':
        return (
          <TabContent>
            <Upload {...fileProps}>
              <UploadArea>
                <div className="upload-icon">
                  <InboxOutlined style={{ filter: 'drop-shadow(0 2px 3px rgba(0,0,0,0.1))' }} />
                </div>
                <div className="upload-text">
                  Drag and drop file, or <span className="browse-link">Browse</span>
                </div>
                <div className="file-types">Supports PDF, TXT, MARKDOWN, Max 15MB each.</div>
              </UploadArea>
            </Upload>
          </TabContent>
        );
      case 'folder':
        return (
          <TabContent>
            <Upload {...folderProps}>
              <UploadArea>
                <div className="upload-icon">
                  <FolderOutlined style={{ filter: 'drop-shadow(0 2px 3px rgba(0,0,0,0.1))' }} />
                </div>
                <div className="upload-text">
                  Drag and drop folder, or <span className="browse-link">Browse</span>
                </div>
                <div className="file-types">Supports TXT, MARKDOWN, PDF. Max 15MB each.</div>
              </UploadArea>
            </Upload>
          </TabContent>
        );
      case 'software':
      case 'wearable':
        return (
          <TabContent>
            <div style={{ textAlign: 'center', color: '#86909C' }}>
              {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} upload coming soon
            </div>
          </TabContent>
        );
      default:
        return null;
    }
  };

  return (
    <>
      <GlobalStyle />
      <div className="p-1">
        <div className="mb-2 text-[15px] font-medium text-gray-700">Upload Method</div>
        <UploadTypeContainer>
          {uploadTypes.map((type) => (
            <UploadTypeBox
              key={type.key}
              active={activeTab === type.key}
              disabled={type.disabled}
              onClick={() => !type.disabled && setActiveTab(type.key)}
            >
              {type.icon}
              <span className="text">{type.text}</span>
            </UploadTypeBox>
          ))}
        </UploadTypeContainer>
        {renderContent()}
      </div>
    </>
  );
}
