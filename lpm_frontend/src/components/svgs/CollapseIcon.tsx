import type React from 'react';
import classNames from 'classnames';

interface CollapseIconProps {
  className?: string;
  collapsed?: boolean;
}

const CollapseIcon: React.FC<CollapseIconProps> = ({ className, collapsed = false }) => {
  return (
    <svg
      className={classNames(
        'w-4 h-4 flex-shrink-0',
        collapsed ? 'rotate-180' : '',
        'transition-transform',
        className
      )}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        d="M4 6h16M4 12h16M4 18h16"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
      />
    </svg>
  );
};

export default CollapseIcon;
