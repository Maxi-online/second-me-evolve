'use client';

import { useEffect, useState } from 'react';
import { Form, Input, Button, message, Modal } from 'antd';
import AddParticipantModal from './AddParticipantModal';
import { useSpaceStore } from '@/store/useSpaceStore';
import { useUploadStore } from '@/store/useUploadStore';

interface CreateSpaceModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (data: {
    name: string;
    objective: string;
    participants: {
      url: string;
      name: string;
      avatar?: string;
      roleDescription?: string;
    }[];
  }) => void;
  currentSecondMe: string;
}

function CreateSpaceModal({ onClose, onSubmit, currentSecondMe, open }: CreateSpaceModalProps) {
  const [form] = Form.useForm();
  const [showAddParticipantModal, setShowAddParticipantModal] = useState(false);
  const [formValid, setFormValid] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const addSpace = useSpaceStore((state) => state.addSpace);
  const startSpace = useSpaceStore((state) => state.startSpace);
  const fetchUploadList = useUploadStore((state) => state.fetchUploadList);

  const init = () => {
    form.resetFields();
    setShowAddParticipantModal(false);
    setFormValid(false);
    setSubmitting(false);
  };

  useEffect(() => {
    if (open) {
      fetchUploadList();
    } else {
      init();
    }
  }, [open]);

  // Monitor form fields changes
  const handleFormChange = () => {
    const values = form.getFieldsValue();
    const isValid = values.name && values.objective;

    setFormValid(isValid);
  };
  // Extract name from URL function
  const extractNameFromUrl = (url: string): string => {
    try {
      const urlParts = url.split('/');

      if (urlParts.length >= 4) {
        return urlParts[3]; // This should be the Second Me name
      }

      return 'Unknown Second Me';
    } catch (error) {
      console.error('Error extracting name from URL:', error);

      return 'Unknown Second Me';
    }
  };

  const [participants, setParticipants] = useState([
    {
      url: currentSecondMe,
      name: extractNameFromUrl(currentSecondMe),
      avatar: `https://api.dicebear.com/7.x/avataaars/svg?seed=${currentSecondMe}`,
      roleDescription: 'Host of this space'
    }
  ]);

  const handleSubmit = () => {
    form.validateFields().then(async (values) => {
      setSubmitting(true);

      // Extract participant URLs and info
      const participantUrls = participants.map((p) => p.url);
      const participantsInfo = participants.map((p) => ({
        url: p.url,
        role_description: p.roleDescription
      }));

      try {
        // Call the API through our store to create space
        const newSpace = await addSpace({
          title: values.name,
          objective: values.objective,
          host: currentSecondMe,
          participants: participantUrls,
          participants_info: participantsInfo
        });

        if (newSpace) {
          // Start the space
          const startResult = await startSpace(newSpace.id);

          if (startResult) {
            message.success('Space created and started successfully!');

            // Open the space in a new tab
            window.open(`/standalone/space/${newSpace.id}`, '_blank');

            // Call the original onSubmit for UI updates
            onSubmit({
              name: values.name,
              objective: values.objective,
              participants
            });
          } else {
            message.warning('Space created but failed to start. Please try starting it manually.');

            // Call the original onSubmit for UI updates
            onSubmit({
              name: values.name,
              objective: values.objective,
              participants
            });
          }
        }
      } catch (error) {
        message.error('Failed to create space. Please try again.');
        console.error('Error creating space:', error);
      } finally {
        setSubmitting(false);
      }
    });
  };

  const addParticipant = (portalUrl: string, roleDescription?: string) => {
    // Extract name from URL
    const displayName = extractNameFromUrl(portalUrl);

    setParticipants([
      ...participants,
      {
        url: portalUrl,
        name: displayName,
        avatar: `https://api.dicebear.com/7.x/avataaars/svg?seed=${displayName}`,
        roleDescription: roleDescription || ''
      }
    ]);
    setShowAddParticipantModal(false);
  };

  const showAddParticipant = () => {
    setShowAddParticipantModal(true);
  };

  const updateParticipant = (index: number, url: string, roleDescription?: string) => {
    const newParticipants = [...participants];
    // Extract name from URL
    const displayName = extractNameFromUrl(url);

    newParticipants[index] = {
      url,
      name: displayName,
      avatar: `https://api.dicebear.com/7.x/avataaars/svg?seed=${displayName}`,
      roleDescription: roleDescription || (index === 0 ? 'Host of this space' : '')
    };
    setParticipants(newParticipants);
  };

  // delete participant
  const removeParticipant = (index: number) => {
    if (index === 0) return;

    const newParticipants = [...participants];

    newParticipants.splice(index, 1);
    setParticipants(newParticipants);
  };

  return (
    <Modal
      footer={[
        <Button key="cancel" onClick={onClose}>
          Cancel
        </Button>,
        <Button
          key="submit"
          disabled={!formValid || submitting}
          loading={submitting}
          onClick={handleSubmit}
          type="primary"
        >
          Create Space
        </Button>
      ]}
      onCancel={onClose}
      open={open}
      title="Create New Collaboration Space"
      width={600}
    >
      <Form
        form={form}
        initialValues={{
          name: '',
          objective: ''
        }}
        layout="vertical"
        onValuesChange={handleFormChange}
      >
        <Form.Item
          label="Space Name"
          name="name"
          rules={[{ required: true, message: 'Please enter a name for your space' }]}
        >
          <Input placeholder="e.g., Market Analysis Space" />
        </Form.Item>

        <Form.Item
          label="Space Task"
          name="objective"
          rules={[{ required: true, message: 'Please enter the objective of this space' }]}
        >
          <Input.TextArea
            placeholder="Describe the task for this collaboration space..."
            rows={4}
          />
        </Form.Item>

        <div className="mb-4">
          <div className="flex justify-between items-center mb-2">
            <label className="text-sm font-medium text-gray-700">Participants</label>
            <Button onClick={showAddParticipant} type="link">
              + Add Participant
            </Button>
            <AddParticipantModal
              onAdd={addParticipant}
              onClose={() => setShowAddParticipantModal(false)}
              open={showAddParticipantModal}
            />
          </div>

          <div className="space-y-3">
            {participants.map((participant, index) => (
              <div key={index} className="flex items-center space-x-3">
                {participant.avatar && (
                  <img
                    alt={participant.name}
                    className="w-8 h-8 rounded-full"
                    src={participant.avatar}
                  />
                )}
                <Input
                  className="flex-1"
                  disabled={index === 0}
                  onChange={(e) => updateParticipant(index, e.target.value)}
                  placeholder="https://app.secondme.io/xxxxx"
                  value={participant.url}
                />
                {participant.name && (
                  <span className="text-sm text-gray-600 min-w-[80px] truncate">
                    {participant.name}
                  </span>
                )}
                {index > 0 && (
                  <button
                    className="text-red-500 hover:text-red-700 transition-colors p-1"
                    onClick={() => removeParticipant(index)}
                    title="Remove participant"
                    type="button"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                      />
                    </svg>
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      </Form>
    </Modal>
  );
}

export default CreateSpaceModal;
