import ReactMD from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { ReactElement } from 'react';

interface CodeProps {
  node?: any;
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
}

const CodeBlock = ({ node, inline, className, children, ...props }: CodeProps): ReactElement => {
  const match = /language-(\w+)/.exec(className || '');

  return !inline && match ? (
    <SyntaxHighlighter PreTag="div" language={match[1]} style={oneDark} {...props}>
      {String(children).replace(/\n$/, '')}
    </SyntaxHighlighter>
  ) : (
    <code className={className} {...props}>
      {children}
    </code>
  );
};

interface IProps {
  markdownContent: string;
  className?: string;
}

const Markdown = (props: IProps): ReactElement => {
  const { markdownContent, className } = props;

  return (
    <ReactMD
      className={className}
      components={{
        code: CodeBlock
      }}
    >
      {markdownContent}
    </ReactMD>
  );
};

export default Markdown;
