'use client';

import RightArrowIcon from '@/components/svgs/RightArrowIcon';
import classNames from 'classnames';
import { useState } from 'react';

interface Tool {
  name: string;
  description: string;
  unreleased?: boolean;
}

interface IProps {
  setChosenTool: (toolName: string) => void;
  chosenTool: string;
}

const tools: Tool[] = [
  {
    name: 'Second Me Chat API',
    description: 'Connect your application with external APIs and services.'
  },
  {
    name: 'Bridge Mode API',
    description: 'Transform data between different formats and structures.',
    unreleased: true
  }
];

const Tools = (props: IProps) => {
  const { setChosenTool, chosenTool } = props;

  const toggleTool = (toolName: string) => {
    setChosenTool(toolName == chosenTool ? '' : toolName);
  };

  return (
    <div className="p-4 flex flex-col gap-4 h-[200px]">
      {tools.map((tool) => {
        const isExpanded = chosenTool == tool.name;

        return (
          <div
            key={tool.name}
            className="border border-gray-200 rounded-lg shadow-sm bg-white overflow-hidden"
          >
            <div
              className="flex justify-between items-center p-4 cursor-pointer"
              onClick={() => (tool.unreleased ? null : toggleTool(tool.name))}
            >
              <div className="font-semibold text-gray-800 flex items-center">
                {tool.name}
                {tool.unreleased && (
                  <span className="ml-2 text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
                    Coming Soon
                  </span>
                )}
              </div>
              <div className={classNames('text-gray-500', isExpanded ? 'rotate-90' : 'rotate-0')}>
                <RightArrowIcon />
              </div>
            </div>

            {isExpanded && (
              <div className="p-1 pl-4 border-t text-gray-600 border-gray-100">
                {tool.description}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default Tools;
