import React from 'react';
import classNames from 'classnames';

interface GlobeIconProps {
  className?: string;
}

const GlobeIcon: React.FC<GlobeIconProps> = ({ className }) => {
  return (
    <svg
      className={classNames('w-4 h-4 flex-shrink-0', className)}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
      />
    </svg>
  );
};

export default GlobeIcon;
