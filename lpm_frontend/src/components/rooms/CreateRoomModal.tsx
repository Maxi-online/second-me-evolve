'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import { Form, Input, Button, message } from 'antd';

const Modal = dynamic(() => import('antd/lib/modal'), {
  ssr: false
});

interface CreateRoomModalProps {
  onClose: () => void;
  onSubmit: (data: { name: string; objective: string; participants: { id: string }[] }) => void;
  currentSecondMeId: string;
}

function CreateRoomModal({ onClose, onSubmit, currentSecondMeId }: CreateRoomModalProps) {
  const [form] = Form.useForm();
  const [participants, setParticipants] = useState([{ id: currentSecondMeId }]);

  const handleSubmit = () => {
    if (participants.length < 2) {
      message.error('A room must have at least 2 participants.');

      return;
    }

    // Verify all participants have valid URLs
    const invalidParticipants = participants.filter(
      (p) => !p.id.startsWith('https://secondme.com/')
    );

    if (invalidParticipants.length > 0) {
      message.error('All participants must have valid SecondMe URLs (https://secondme.com/xxxxx)');

      return;
    }

    form.validateFields().then((values) => {
      onSubmit({
        name: values.name,
        objective: values.objective,
        participants
      });
    });
  };

  const addParticipant = () => {
    setParticipants([...participants, { id: '' }]);
  };

  const updateParticipant = (index: number, id: string) => {
    const newParticipants = [...participants];

    newParticipants[index] = { id };
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
          disabled={participants.length < 2}
          onClick={handleSubmit}
          type="primary"
        >
          Create Room
        </Button>
      ]}
      onCancel={onClose}
      open={true}
      title="Create New Room"
      width={600}
    >
      <Form
        form={form}
        initialValues={{
          name: '',
          description: ''
        }}
        layout="vertical"
      >
        <Form.Item
          label="Room Name"
          name="name"
          rules={[{ required: true, message: 'Please enter a room name' }]}
        >
          <Input placeholder="Enter room name" />
        </Form.Item>

        <Form.Item
          label="Room Objective"
          name="objective"
          rules={[{ required: true, message: 'Please enter the room objective' }]}
        >
          <Input.TextArea placeholder="What do you want to achieve in this room?" rows={4} />
        </Form.Item>

        <div className="mb-4">
          <div className="flex justify-between items-center mb-2">
            <div>
              <h4 className="text-base font-medium">Participants</h4>
              <p className="text-sm text-gray-500">At least 2 participants required</p>
            </div>
            <Button onClick={addParticipant} type="link">
              Add Participant
            </Button>
          </div>

          {participants.map((participant, index) => (
            <div key={index} className="mb-4 p-4 border border-gray-200 rounded-lg">
              <div className="mb-3">
                <label className="block text-sm font-medium text-gray-700 mb-1">SecondMe URL</label>
                <div className="text-xs text-gray-500 mb-2">
                  Enter the full URL of the SecondMe instance (e.g., https://secondme.com/23581)
                </div>
                <Input
                  disabled={index === 0}
                  onChange={(e) => updateParticipant(index, e.target.value)}
                  placeholder="https://secondme.com/xxxxx"
                  value={participant.id}
                />
              </div>
            </div>
          ))}
        </div>
      </Form>
    </Modal>
  );
}

export default dynamic(() => Promise.resolve(CreateRoomModal), {
  ssr: false
});
