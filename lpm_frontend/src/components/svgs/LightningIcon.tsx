import React from 'react';
import classNames from 'classnames';

interface LightningIconProps {
  className?: string;
}

const LightningIcon: React.FC<LightningIconProps> = ({ className }) => {
  return (
    <svg
      className={classNames('w-4 h-4', className)}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        d="M13 10V3L4 14h7v7l9-11h-7z"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
      />
    </svg>
  );
};

export default LightningIcon;
