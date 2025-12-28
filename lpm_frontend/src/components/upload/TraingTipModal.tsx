import { Modal } from 'antd';

interface IProps {
  open: boolean;
  onClose: () => void;
  confirm: () => void;
}

const TrainingTipModal = (props: IProps) => {
  const { open, onClose: handleCancel, confirm: handleConfirm } = props;

  return (
    <Modal
      okText="Continue Start"
      onCancel={handleCancel}
      onOk={handleConfirm}
      open={open}
      title="Training Tip"
    >
      Turning on services during training may result in runtime model replacement
    </Modal>
  );
};

export default TrainingTipModal;
