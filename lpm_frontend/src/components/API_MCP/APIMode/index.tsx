'use client';

import { Button, Space } from 'antd';
import { useState } from 'react';
import SecondMeChatAPI from './components/SecondMeChatAPI';
import BridgeModeAPI from './components/BridgeModeAPI';
import classNames from 'classnames';

const APIMode = () => {
  const [activeTab, setActiveTab] = useState('1');

  const handleTabChange = (key: string) => {
    setActiveTab(key);
  };

  return (
    <div className="w-full">
      <div className="mb-6">
        <Space className="flex justify-center sm:justify-start" size="middle">
          <Button
            className={classNames(
              'min-w-[180px] h-10 flex items-center justify-center',
              activeTab === '1' && '!bg-blue-50 !text-blue-600'
            )}
            onClick={() => handleTabChange('1')}
          >
            Second Me Chat API
          </Button>
          <Button
            className={classNames(
              'min-w-[180px] h-10 flex items-center justify-center',
              activeTab === '2' && '!bg-blue-50 !text-blue-600'
            )}
            onClick={() => handleTabChange('2')}
          >
            <span>Bridge Mode API</span>
          </Button>
        </Space>
      </div>

      <div className="mt-4">
        {activeTab === '1' && <SecondMeChatAPI />}
        {activeTab === '2' && <BridgeModeAPI />}
      </div>
    </div>
  );
};

export default APIMode;
