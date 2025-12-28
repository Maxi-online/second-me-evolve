import MCPMode from '@/components/API_MCP/MCPMode';
import APIMode from '@/components/API_MCP/APIMode';
import { Tabs } from 'antd';
import styles from '@/components/API_MCP/index.module.css';
import classNames from 'classnames';

const items = [
  {
    key: 'api',
    label: 'API',
    children: <APIMode />
  },
  {
    key: 'mcp',
    label: 'MCP',
    children: <MCPMode />
  }
];

const Page = () => {
  return (
    <div
      className={classNames(
        'w-full h-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6',
        styles.container
      )}
    >
      <div className="flex flex-col mb-10">
        <div className="font-extrabold text-2xl">API & MCP</div>
        <div>Manage your API and MCP services</div>
      </div>
      <Tabs className="w-full" defaultActiveKey="api" items={items} />
    </div>
  );
};

export default Page;
