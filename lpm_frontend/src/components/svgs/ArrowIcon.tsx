import type React from 'react';
import classNames from 'classnames';

interface ArrowIconProps {
  className?: string;
}

const ArrowIcon: React.FC<ArrowIconProps> = ({ className }) => {
  return (
    <svg
      className={classNames('w-4 h-4 flex-shrink-0 transition-transform', className)}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path d="M9 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} />
    </svg>
  );
};

export default ArrowIcon;
