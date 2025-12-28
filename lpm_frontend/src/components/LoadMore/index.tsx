import { Spin } from 'antd';
import classNames from 'classnames';
import { memo, useEffect, useRef } from 'react';

interface IProps {
  className?: string;
  scrollContainerId?: string;
  loadMore: () => Promise<void>;
}

function LoadMore(props: IProps): JSX.Element {
  const { loadMore, className, scrollContainerId = '#scrollContainer' } = props;
  const eleRef = useRef<HTMLDivElement>(null);
  const loadingRef = useRef<boolean>(false);

  useEffect(() => {
    if (!eleRef.current) {
      return;
    }

    function callback(entries: IntersectionObserverEntry[]) {
      entries.forEach((entry) => {
        if (entry.isIntersecting && !loadingRef.current) {
          loadingRef.current = true;

          loadMore().finally(() => {
            loadingRef.current = false;
          });
        }
      });
    }

    const observer = new IntersectionObserver(callback, {
      rootMargin: '50px',
      root: document.querySelector(scrollContainerId),
      threshold: [0.1]
    });

    const delayInMilliseconds = 500;
    const timeoutId = setTimeout(() => {
      if (eleRef.current) {
        observer.observe(eleRef.current);
      }
    }, delayInMilliseconds);

    return () => {
      clearTimeout(timeoutId);
      observer.disconnect();
    };
  }, [loadMore, scrollContainerId]);

  return (
    <div ref={eleRef} className={classNames('flex items-center justify-center', className)}>
      <Spin />
    </div>
  );
}

export default memo(LoadMore);
