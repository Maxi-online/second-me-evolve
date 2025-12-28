import type { SVGProps } from 'react';
import classNames from 'classnames';

interface DocumentIconProps extends SVGProps<SVGSVGElement> {
  className?: string;
}

const DocumentIcon = ({ className = '', ...props }: DocumentIconProps): JSX.Element => {
  return (
    <svg
      className={classNames('w-4 h-4 shrink-0', className)}
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" x2="8" y1="13" y2="13" />
      <line x1="16" x2="8" y1="17" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  );
};

export default DocumentIcon;
