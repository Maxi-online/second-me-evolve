'use client';

import { CalendarOutlined } from '@ant-design/icons';

const BridgeModeAPI = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-[50vh] p-4">
      <div className="text-center">
        <div className="flex justify-center mb-4">
          <CalendarOutlined style={{ fontSize: '48px', color: '#1677ff' }} />
        </div>
        <h2 className="text-xl font-bold mb-3">Coming Soon</h2>
        <p className="text-gray-600">
          We are working hard to bring you Bridge Mode API features. Stay tuned!
        </p>
      </div>
    </div>
  );
};

export default BridgeModeAPI;
