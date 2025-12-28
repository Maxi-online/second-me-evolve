import React from 'react';
import classNames from 'classnames';

interface UserIconProps {
  className?: string;
}

const UserIcon: React.FC<UserIconProps> = ({ className }) => {
  return (
    <svg
      className={classNames('w-4 h-4 flex-shrink-0', className)}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
      />
    </svg>
  );
};

export default UserIcon;
