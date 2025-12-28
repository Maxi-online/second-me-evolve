import classNames from 'classnames';
import VideoIcon from '@/components/svgs/VideoIcon';
import ChatBubbleIcon from '@/components/svgs/ChatBubbleIcon';
import LightningIcon from '@/components/svgs/LightningIcon';
import UsersIcon from '@/components/svgs/UsersIcon';

interface IProps {
  className?: string;
}

const Footer = (props: IProps) => {
  const { className } = props;

  return (
    <div
      className={classNames(
        `fixed bottom-0 left-0 right-0 py-4 transition-opacity delay-700 duration-700 ease-in-out`,
        className
      )}
    >
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row items-center justify-center gap-3">
          <span className="font-medium text-secondme-navy text-sm">See Demos:</span>
          <div className="flex flex-wrap justify-center gap-x-4 gap-y-2">
            <a
              className="text-sm text-secondme-blue hover:text-secondme-blue/80 hover:underline flex items-center gap-1"
              href="https://youtu.be/TogHaqdExvc"
              rel="noopener noreferrer"
              target="_blank"
            >
              <VideoIcon />
              Walkthrough Video
            </a>
            <a
              className="text-sm text-secondme-blue hover:text-secondme-blue/80 hover:underline flex items-center gap-1"
              href="https://app.secondme.io/example/ama"
              rel="noopener noreferrer"
              target="_blank"
            >
              <ChatBubbleIcon />
              Felix AMA (Roleplay)
            </a>
            <a
              className="text-sm text-secondme-blue hover:text-secondme-blue/80 hover:underline flex items-center gap-1"
              href="https://app.secondme.io/example/brainstorming"
              rel="noopener noreferrer"
              target="_blank"
            >
              <LightningIcon />
              Brainstorming (Network)
            </a>
            <a
              className="text-sm text-secondme-blue hover:text-secondme-blue/80 hover:underline flex items-center gap-1"
              href="https://app.secondme.io/example/Icebreaker"
              rel="noopener noreferrer"
              target="_blank"
            >
              <UsersIcon />
              Icebreaker (Network)
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Footer;
