import React from 'react';
import classNames from 'classnames';

interface UploadIconProps {
  className?: string;
}

const UploadIcon: React.FC<UploadIconProps> = ({ className }) => {
  return (
    <svg
      className={classNames('w-4 h-4 flex-shrink-0', className)}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
      />
    </svg>
  );
};

export default UploadIcon;
