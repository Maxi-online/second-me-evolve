import type React from 'react';
import classNames from 'classnames';

interface BridgeIconProps {
  className?: string;
}

const BridgeIcon: React.FC<BridgeIconProps> = ({ className }) => {
  return (
    <svg
      className={classNames('w-4 h-4 flex-shrink-0', className)}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
      />
    </svg>
  );
};

export default BridgeIcon;
