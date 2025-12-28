'use client';

import { useMemo, useRef, useState } from 'react';
import { Modal, Form, Input, Button, Select, Spin } from 'antd';
import { useUploadStore } from '@/store/useUploadStore';

const { Option } = Select;

interface AddParticipantModalProps {
  open: boolean;
  onClose: () => void;
  onAdd: (portalUrl: string, roleDescription?: string) => void;
}

export default function AddParticipantModal({ open, onClose, onAdd }: AddParticipantModalProps) {
  const [form] = Form.useForm();
  const [selectForm] = Form.useForm();
  const uploads = useUploadStore((state) => state.uploads);
  const total = useUploadStore((state) => state.total);
  const [showSelectModal, setShowSelectModal] = useState(false);
  const fetchUploadList = useUploadStore((state) => state.fetchUploadList);
  const loadMoreRef = useRef(false);

  const handleSubmit = () => {
    form.validateFields().then((values) => {
      onAdd(values.portalUrl, values.roleDescription);
      form.resetFields();
      onClose();
    });
  };

  const handleSecondMeSelect = (value: string) => {
    const [secondMeName, instanceId] = value.split('|');
    const url = `https://app.secondme.io/${secondMeName}/${instanceId}`;

    form.setFieldsValue({ portalUrl: url });
    setShowSelectModal(false);
    selectForm.resetFields();
  };

  const handleConfirmSelect = () => {
    selectForm.validateFields().then((values) => {
      if (values.selectedSecondMe) {
        handleSecondMeSelect(values.selectedSecondMe);
      } else {
        setShowSelectModal(false);
      }
    });
  };

  const handlePopupScroll = (e: any) => {
    const { target } = e;

    if (target.scrollTop + target.offsetHeight >= target.scrollHeight - 20) {
      if (!loadMoreRef.current && uploads.length < total) {
        loadMoreRef.current = true;

        fetchUploadList(false).finally(() => {
          loadMoreRef.current = false;
        });
      }
    }
  };

  const RenderOptions = useMemo(() => {
    const registeredUploadId = JSON.parse(localStorage.getItem('registeredUpload')!)?.instance_id;

    return uploads.map((upload) => {
      if (upload.instance_id == registeredUploadId) {
        return null;
      }

      return (
        <Option
          key={`${upload.upload_name}|${upload.instance_id}`}
          disabled={upload.status !== 'online'}
          value={`${upload.upload_name}|${upload.instance_id}`}
        >
          <div className="flex items-center justify-between">
            <span>{upload.upload_name}</span>
            <div
              className={`px-2 py-0.5 text-xs rounded-full flex items-center gap-1 ${
                upload.status === 'offline'
                  ? 'bg-gray-100 text-gray-800'
                  : upload.status === 'registered'
                    ? 'bg-yellow-100 text-yellow-800'
                    : 'bg-green-100 text-green-800'
              }`}
            >
              <div
                className={`w-1.5 h-1.5 rounded-full ${
                  upload.status === 'offline'
                    ? 'bg-gray-500'
                    : upload.status === 'registered'
                      ? 'bg-yellow-600'
                      : 'bg-green-600'
                }`}
              />
              {upload.status || 'Unknown'}
            </div>
          </div>
        </Option>
      );
    });
  }, [uploads]);

  return (
    <Modal
      footer={[
        <Button key="cancel" onClick={onClose}>
          Cancel
        </Button>,
        <Button key="submit" onClick={handleSubmit} type="primary">
          Connect and Add
        </Button>
      ]}
      onCancel={onClose}
      open={open}
      title="Add Second Me as Participant"
    >
      <Form
        form={form}
        initialValues={{
          portalUrl: '',
          roleDescription: ''
        }}
        layout="vertical"
      >
        <div className="mb-2">
          <Form.Item
            label="Second Me URL"
            name="portalUrl"
            rules={[{ required: true, message: 'Please enter the Second Me URL' }]}
          >
            <Input placeholder="Enter Second Me URL" />
          </Form.Item>
          <div className="flex justify-end -mt-2 mb-2">
            <span
              className="text-blue-500 text-xs cursor-pointer"
              onClick={() => setShowSelectModal(true)}
            >
              Choose from registered Second Me
            </span>
          </div>
        </div>

        {/* Role Description hiding */}
        <Form.Item hidden name="roleDescription">
          <Input.TextArea rows={3} />
        </Form.Item>
      </Form>

      {/* Selection Modal */}
      <Modal
        footer={[
          <Button key="cancel" onClick={() => setShowSelectModal(false)}>
            Cancel
          </Button>,
          <Button key="confirm" onClick={handleConfirmSelect} type="primary">
            Confirm
          </Button>
        ]}
        onCancel={() => setShowSelectModal(false)}
        open={showSelectModal}
        title="Select Registered Second Me"
      >
        <Form form={selectForm} layout="vertical">
          <Form.Item
            name="selectedSecondMe"
            rules={[{ required: true, message: 'Please select a Second Me' }]}
          >
            <Select
              onPopupScroll={handlePopupScroll}
              placeholder="Select a registered Second Me"
              size="large"
              style={{ width: '100%' }}
            >
              {RenderOptions}
              {uploads.length < total && (
                <Option key="loading" disabled value="loading">
                  <div className="flex justify-center py-2">
                    <Spin className="my-4" size="small" />
                  </div>
                </Option>
              )}
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </Modal>
  );
}
