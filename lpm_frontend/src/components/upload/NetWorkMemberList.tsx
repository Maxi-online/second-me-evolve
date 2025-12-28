import { Spin } from 'antd';
import { useCallback, useEffect } from 'react';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  GlobalOutlined,
  DoubleRightOutlined,
  SyncOutlined
} from '@ant-design/icons';
import { useUploadStore } from '@/store/useUploadStore';
import { useLoadInfoStore } from '@/store/useLoadInfoStore';
import classNames from 'classnames';
import LoadMore from '../LoadMore';

const colorClasses = [
  'bg-red-500',
  'bg-pink-500',
  'bg-purple-500',
  'bg-indigo-500',
  'bg-blue-500',
  'bg-cyan-500',
  'bg-teal-500',
  'bg-green-500',
  'bg-lime-500',
  'bg-yellow-500',
  'bg-amber-500',
  'bg-orange-500'
];

const NetWorkMemberList = () => {
  const loading = useUploadStore((state) => state.loading);
  const total = useUploadStore((state) => state.total);
  const uploads = useUploadStore((state) => state.uploads);
  const fetchUploadList = useUploadStore((state) => state.fetchUploadList);
  const loadInfo = useLoadInfoStore((state) => state.loadInfo);

  useEffect(() => {
    fetchUploadList();
  }, []);

  const getRandomColor = (str: string) => {
    if (!str) return 'bg-gray-400';

    let hash = 0;

    for (let i = 0; i < str.length; i++) {
      hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }

    const index = Math.abs(hash) % colorClasses.length;

    return colorClasses[index];
  };

  const renderUploadList = useCallback(() => {
    return uploads?.map((upload) => {
      const uploadStatus =
        upload.instance_id == loadInfo?.instance_id ? loadInfo?.status : upload?.status;

      return (
        <div
          key={upload?.instance_id}
          className={classNames(
            'flex items-center justify-between px-4 py-3 rounded-lg shadow-sm transition-all duration-200 hover:shadow-md border ',
            uploadStatus === 'online' ? 'bg-blue-50 border-blue-100' : 'bg-gray-50 border-gray-200',
            'group'
          )}
          onClick={() => {
            window.open(
              `https://app.secondme.io/${upload.upload_name}/${upload.instance_id}`,
              '_blank'
            );
          }}
        >
          <div className="flex items-center gap-3 w-full">
            <div
              className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center text-white ${getRandomColor(upload?.upload_name)}`}
            >
              {upload?.upload_name ? upload?.upload_name.charAt(0).toUpperCase() : '?'}
            </div>
            <div>
              <div className={`font-medium text-gray-700`}>{upload?.upload_name || 'Unknown'}</div>
              <div className="flex items-center gap-2 mt-1">
                <div
                  className={`px-2 py-0.5 text-xs rounded-full flex items-center gap-1.5 ${
                    uploadStatus === 'offline'
                      ? 'bg-gray-100 text-gray-800'
                      : uploadStatus === 'registered'
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-green-100 text-green-800'
                  }`}
                >
                  {uploadStatus === 'offline' ? (
                    <CloseCircleOutlined className="text-gray-500" />
                  ) : uploadStatus === 'registered' ? (
                    <SyncOutlined className="text-yellow-600" spin />
                  ) : (
                    <CheckCircleOutlined className="text-green-600" />
                  )}
                  {uploadStatus || 'Unknown'}
                </div>
              </div>
            </div>
          </div>

          <DoubleRightOutlined
            className="text-[12px] !text-blue-500 transition-colors"
            style={{ fontSize: '12px' }}
          />
        </div>
      );
    });
  }, [uploads, loadInfo]);

  if (loading && uploads.length == 0) {
    return <Spin className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />;
  }

  if (uploads.length === 0) {
    return (
      <div className="text-gray-500 text-sm p-8 bg-gray-50 rounded-lg border border-gray-200 text-center shadow-sm">
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center text-gray-400">
            <GlobalOutlined style={{ fontSize: '24px' }} />
          </div>
        </div>
        <p className="text-gray-600 font-medium">Network is Empty</p>
        <p className="text-sm text-gray-400 mt-2">Be the first to join the Second Me network!</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full gap-2">
      {renderUploadList()}
      {total !== uploads.length && (
        <LoadMore
          className="my-5"
          loadMore={() => fetchUploadList(false)}
          scrollContainerId="#netWorkMemberScrollList"
        />
      )}
    </div>
  );
};

export default NetWorkMemberList;
