import type React from 'react';
import classNames from 'classnames';

interface DoneIconProps {
  className?: string;
}

const DoneIcon: React.FC<DoneIconProps> = ({ className }) => {
  return (
    <svg className={classNames('w-4 h-4', className)} fill="currentColor" viewBox="0 0 20 20">
      <path
        clipRule="evenodd"
        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
        fillRule="evenodd"
      />
    </svg>
  );
};

export default DoneIcon;
