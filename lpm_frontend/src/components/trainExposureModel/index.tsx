import type { TrainStepOutput } from '@/service/train';
import { getStepOutputContent } from '@/service/train';
import { Modal, Table } from 'antd';
import { useEffect, useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/cjs/styles/prism';

export interface IStepOutputInfo {
  path?: string;
  stepName: string;
}
interface IProps {
  handleClose: () => void;
  stepOutputInfo?: IStepOutputInfo;
}

const TrainExposureModel = (props: IProps) => {
  const { handleClose, stepOutputInfo } = props;
  const [outputContent, setOutputContent] = useState<TrainStepOutput | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    if (!stepOutputInfo?.stepName) return;

    setOutputContent(null);
    setLoading(true);

    getStepOutputContent(stepOutputInfo.stepName)
      .then((res) => {
        if (res.data.code == 0) {
          const data = res.data.data;

          setOutputContent(data);
        } else {
          console.error(res.data.message);
        }
      })
      .finally(() => {
        setLoading(false);
      });
  }, [stepOutputInfo?.stepName]);

  const renderOutputContent = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center w-full py-12">
          <div className="flex flex-col items-center space-y-4">
            <div className="relative w-12 h-12">
              <div className="absolute w-12 h-12 rounded-full border-2 border-gray-200" />
              <div
                className="absolute w-12 h-12 rounded-full border-2 border-t-blue-500 animate-spin"
                style={{ animationDuration: '1.2s' }}
              />
            </div>
            <p className="text-gray-500 text-sm">loading...</p>
          </div>
        </div>
      );
    }

    if (!outputContent) return 'There are no resources for this step at this time';

    if (outputContent.file_type == 'json') {
      const showContent = JSON.stringify(outputContent.content, null, 2);

      return (
        <SyntaxHighlighter
          customStyle={{
            backgroundColor: 'transparent',
            margin: 0,
            padding: 0
          }}
          language="json"
          style={tomorrow}
        >
          {showContent}
        </SyntaxHighlighter>
      );
    }

    if (outputContent.file_type == 'parquet') {
      const columns = outputContent.columns.map((item, index) => ({
        title: item,
        dataIndex: item,
        key: index
      }));
      const data = outputContent.content;

      return (
        <Table className="w-fit max-w-fit" columns={columns} dataSource={data} pagination={false} />
      );
    }

    return 'There are no resources for this step at this time';
  };

  return (
    <Modal
      centered
      closable={false}
      footer={null}
      onCancel={handleClose}
      open={!!stepOutputInfo?.stepName}
      width={800}
    >
      <div className="flex flex-col">
        {stepOutputInfo?.path && (
          <div className="flex items-center justify-between mb-4">
            <span className="text-lg font-medium text-gray-900">{`path: ${stepOutputInfo.path}`}</span>
          </div>
        )}
        <div className="bg-[#f5f5f5] flex flex-col max-h-[600px] w-full overflow-scroll border border-[#e0e0e0] rounded p-4 font-mono text-sm leading-6 text-[#333] shadow-sm transition-all duration-300 ease-in-out">
          {renderOutputContent()}
        </div>
      </div>
    </Modal>
  );
};

export default TrainExposureModel;
