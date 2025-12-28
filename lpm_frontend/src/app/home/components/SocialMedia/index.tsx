import classNames from 'classnames';
import TwitterXIcon from '@/components/svgs/TwitterXIcon';
import DiscordIcon from '@/components/svgs/DiscordIcon';

interface IProps {
  className?: string;
}

const SocialMedia = (props: IProps) => {
  const { className } = props;

  return (
    <div
      className={classNames(
        `fixed bottom-4 right-4 flex items-center gap-3 transition-opacity duration-700 delay-[800ms] ease-in-out}`,
        className
      )}
    >
      <a
        aria-label="Follow us on X (Twitter)"
        className="w-10 h-10 flex items-center justify-center rounded-lg bg-black text-white hover:bg-gray-800 transition-colors"
        href="https://x.com/SecondMe_AI1"
        rel="noopener noreferrer"
        target="_blank"
      >
        <TwitterXIcon className="w-5 h-5" />
      </a>
      <a
        aria-label="Join our Discord server"
        className="w-10 h-10 flex items-center justify-center rounded-lg bg-[#5865F2] text-white hover:bg-[#4a57e0] transition-colors"
        href="https://discord.com/invite/GpWHQNUwrg"
        rel="noopener noreferrer"
        target="_blank"
      >
        <DiscordIcon className="w-5 h-5" />
      </a>
    </div>
  );
};

export default SocialMedia;
