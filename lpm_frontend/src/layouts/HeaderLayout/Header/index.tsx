'use client';

import { usePathname, useRouter } from 'next/navigation';
import Link from 'next/link';
import Image from 'next/image';
import { GithubOutlined, FileTextOutlined } from '@ant-design/icons';
import { ModelStatus } from '../../../components/ModelStatus';
import { useLoadInfoStore } from '@/store/useLoadInfoStore';
import { useEffect } from 'react';
import { useUploadStore } from '@/store/useUploadStore';
import { ROUTER_PATH } from '@/utils/router';
import { EVENT } from '@/utils/event';
import GitHubStars from '@/components/GithubStars';

export function Header() {
  const pathname = usePathname();
  const isHomePage = pathname === ROUTER_PATH.HOME;
  const isWhitelistPage = pathname.startsWith(ROUTER_PATH.STANDALONE);
  const fetchLoadInfo = useLoadInfoStore((state) => state.fetchLoadInfo);
  const fetchUploadList = useUploadStore((state) => state.fetchUploadList);

  const router = useRouter();

  useEffect(() => {
    fetchLoadInfo();
    fetchUploadList();
  }, []);

  useEffect(() => {
    const handleLogout = () => {
      localStorage.clear();
      router.replace(ROUTER_PATH.HOME);
    };

    addEventListener(EVENT.LOGOUT, handleLogout);

    return () => {
      removeEventListener(EVENT.LOGOUT, handleLogout);
    };
  }, []);

  return (
    <header
      className={`sticky top-0 z-50 w-full overflow-scroll ${
        isHomePage
          ? 'bg-transparent border-none'
          : 'border-b border-gray-800/5 bg-white/60 backdrop-blur-sm'
      }`}
    >
      {!isHomePage && (
        <div className="absolute inset-x-0 -bottom-px h-px bg-gradient-to-r from-transparent via-blue-200/30 to-transparent" />
      )}

      <div className="flex h-16 items-center px-4 sm:px-6 lg:px-8">
        {/* Left section - Logo */}
        <div className="flex-none">
          <Link className="flex items-center space-x-2" href="/">
            <Image
              alt="SecondMe Logo"
              className="object-contain"
              height={40}
              src="/images/logo.png"
              width={160}
            />
          </Link>
        </div>

        {/* Center section - ModelStatus */}
        <div className="flex-1 flex justify-center">
          {!isWhitelistPage && !isHomePage && <ModelStatus />}
        </div>

        {/* Right section - Navigation links */}
        <div className="flex-none flex items-center space-x-6">
          <Link
            className={`flex items-center space-x-1.5 text-sm ${
              isHomePage
                ? 'text-gray-600/90 hover:text-gray-900'
                : 'text-gray-600 hover:text-blue-600'
            } transition-all hover:-translate-y-0.5`}
            href="https://secondme.io"
            rel="noopener noreferrer"
            target="_blank"
            title="Learn about Second Me"
          >
            <div className="flex items-center space-x-1">
              <FileTextOutlined className="text-lg" />
              <span>Whitepaper</span>
            </div>
          </Link>
          <Link
            className={`flex items-center space-x-1.5 text-sm ${
              isHomePage
                ? 'text-gray-600/90 hover:text-gray-900'
                : 'text-gray-600 hover:text-blue-600'
            } transition-all hover:-translate-y-0.5`}
            href="https://github.com/mindverse/Second-Me"
            rel="noopener noreferrer"
            target="_blank"
          >
            <div className="flex items-center space-x-1">
              <GitHubStars />
            </div>
          </Link>
        </div>
      </div>
    </header>
  );
}
