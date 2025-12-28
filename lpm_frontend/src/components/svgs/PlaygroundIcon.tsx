import React from 'react';
import classNames from 'classnames';

interface PlaygroundIconProps {
  className?: string;
}

const PlaygroundIcon: React.FC<PlaygroundIconProps> = ({ className }) => {
  return (
    <svg
      className={classNames('w-5 h-5 flex-shrink-0', className)}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
      />
    </svg>
  );
};

export default PlaygroundIcon;
