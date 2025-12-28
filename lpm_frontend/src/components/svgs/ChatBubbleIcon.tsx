import type React from 'react';
import classNames from 'classnames';

interface ChatBubbleIconProps {
  className?: string;
}

const ChatBubbleIcon: React.FC<ChatBubbleIconProps> = ({ className }) => {
  return (
    <svg
      className={classNames('w-4 h-4', className)}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
      />
    </svg>
  );
};

export default ChatBubbleIcon;
