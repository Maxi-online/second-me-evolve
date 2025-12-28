import type React from 'react';
import classNames from 'classnames';

interface RightArrowIconProps {
  className?: string;
}

const RightArrowIcon: React.FC<RightArrowIconProps> = ({ className }) => {
  return (
    <svg
      className={classNames('h-5 w-5', className)}
      fill="currentColor"
      viewBox="0 0 20 20"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        clipRule="evenodd"
        d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
        fillRule="evenodd"
      />
    </svg>
  );
};

export default RightArrowIcon;
