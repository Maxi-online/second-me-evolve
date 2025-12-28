import { CalendarOutlined } from '@ant-design/icons';

const Page = () => {
  return (
    <div className="w-full h-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <div className="flex flex-col mb-10">
        <div className="font-extrabold text-2xl">Integrations</div>
        <div>Manage your third-party integrations and services</div>
      </div>
      <div className="border-b border-gray-200" />
      <div className="flex flex-col items-center justify-center h-full">
        <div className="text-center">
          <div className="flex justify-center mb-6">
            <CalendarOutlined style={{ fontSize: '64px', color: '#1677ff' }} />
          </div>
          <h1 className="text-3xl font-bold mb-4">Coming Soon</h1>
          <p className="text-lg text-gray-600 mb-6">
            We&apos;re working hard to bring you amazing integration features.
          </p>
          <p className="text-md text-gray-500">Stay tuned for updates!</p>
        </div>
      </div>
    </div>
  );
};

export default Page;
