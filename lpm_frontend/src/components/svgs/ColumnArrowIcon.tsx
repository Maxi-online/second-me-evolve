import type React from 'react';
import classNames from 'classnames';

interface ColumnArrowIconProps {
  className?: string;
}

const ColumnArrowIcon: React.FC<ColumnArrowIconProps> = ({ className }) => {
  return (
    <svg
      className={classNames('w-4 h-4', className)}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 20 20"
    >
      <path
        d="M7 7l3-3 3 3m0 6l-3 3-3-3"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
};

export default ColumnArrowIcon;
