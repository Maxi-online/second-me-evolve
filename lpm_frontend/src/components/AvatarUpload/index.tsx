'use client';

import { useState, useEffect } from 'react';
import { Modal, Upload, message } from 'antd';
import type { RcFile, UploadFile } from 'antd/es/upload';
import type { UploadChangeParam } from 'antd/es/upload/interface';
import { PlusOutlined } from '@ant-design/icons';
import { uploadLoadAvatar } from '@/service/info';
import { useLoadInfoStore } from '@/store/useLoadInfoStore';

interface AvatarUploadProps {
  open: boolean;
  onClose: () => void;
  onAvatarChange: (avatarUrl: string) => void;
  currentAvatar?: string;
}

export default function AvatarUpload({
  open,
  onClose,
  onAvatarChange,
  currentAvatar
}: AvatarUploadProps) {
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewImage, setPreviewImage] = useState('');
  const [fileList, setFileList] = useState<any[]>([]);
  const [messageApi, contextHolder] = message.useMessage();
  const { loadInfo } = useLoadInfoStore();

  const beforeUpload = (file: RcFile) => {
    if (!file) return false;

    const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png';

    if (!isJpgOrPng) {
      message.error('You can only upload JPG/PNG file!');

      return false;
    }

    const isLt2M = file.size / 1024 / 1024 < 2;

    if (!isLt2M) {
      message.error('Image must smaller than 2MB!');

      return false;
    }

    return true;
  };

  const handlePreview = async (file: UploadFile) => {
    if (file.originFileObj) {
      const reader = new FileReader();

      reader.onload = () => {
        setPreviewImage(reader.result as string);
        setPreviewOpen(true);
      };
      reader.readAsDataURL(file.originFileObj);
    } else if (file.url) {
      setPreviewImage(file.url);
      setPreviewOpen(true);
    }
  };

  useEffect(() => {
    if (currentAvatar) {
      setFileList([
        {
          uid: '-1',
          name: 'current-avatar.png',
          status: 'done',
          url: currentAvatar
        }
      ]);
    } else {
      setFileList([]);
    }
  }, [currentAvatar]);

  const handleChange = async (info: UploadChangeParam<UploadFile>) => {
    const { status } = info.file;

    if (status === 'uploading') {
      setFileList([{ ...info.file }]);

      return;
    }

    if (status === 'done' && info.file.originFileObj) {
      const file = info.file.originFileObj;

      // Convert file to base64
      const reader = new FileReader();

      reader.readAsDataURL(file);
      reader.onload = async () => {
        const base64Url = reader.result as string;

        try {
          // Directly use base64 string for upload, no longer using FormData
          if (loadInfo) {
            const res = await uploadLoadAvatar(loadInfo.name, {
              avatar_data: base64Url
            });

            if (res.data.code === 0) {
              // Use base64 image as avatar
              setFileList([
                {
                  uid: '-1',
                  name: file.name,
                  status: 'done',
                  url: base64Url
                }
              ]);

              onAvatarChange(base64Url);

              useLoadInfoStore.getState().fetchLoadInfo();

              messageApi.success('Avatar uploaded successfully');
            } else {
              setFileList([]);
              messageApi.error(res.data.message);
            }
          }
        } catch (err: any) {
          setFileList([]);
          messageApi.error(err.message || 'Failed to upload avatar');
        }
      };
    }
  };

  useEffect(() => {
    return () => {
      fileList.forEach((file) => {
        if (file.url?.startsWith('blob:')) {
          URL.revokeObjectURL(file.url);
        }
      });
    };
  }, [fileList]);

  return (
    <Modal destroyOnClose footer={null} onCancel={onClose} open={open} title="Upload Avatar">
      <div className="p-4">
        {contextHolder}
        <Upload
          accept="image/png,image/jpeg"
          beforeUpload={beforeUpload}
          fileList={fileList}
          listType="picture-card"
          maxCount={1}
          onChange={handleChange}
          onPreview={handlePreview}
          onRemove={() => {
            setFileList([]);
            onAvatarChange('');
          }}
          showUploadList={{
            showPreviewIcon: true,
            showRemoveIcon: true,
            showDownloadIcon: false
          }}
        >
          {fileList.length >= 1 ? null : (
            <div>
              <PlusOutlined />
              <div style={{ marginTop: 8 }}>Upload</div>
            </div>
          )}
        </Upload>
      </div>
      <Modal footer={null} onCancel={() => setPreviewOpen(false)} open={previewOpen}>
        <img alt="Preview" src={previewImage} style={{ width: '100%' }} />
      </Modal>
    </Modal>
  );
}
