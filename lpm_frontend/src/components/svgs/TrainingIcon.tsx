import React from 'react';
import classNames from 'classnames';

interface TrainingIconProps {
  className?: string;
}

const TrainingIcon: React.FC<TrainingIconProps> = ({ className }) => {
  return (
    <svg
      className={classNames('w-4 h-4 flex-shrink-0', className)}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2V6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
      />
    </svg>
  );
};

export default TrainingIcon;
