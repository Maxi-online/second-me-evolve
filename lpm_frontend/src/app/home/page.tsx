'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import CreateSecondMe from '@/app/home/components/Create';
import dynamic from 'next/dynamic';
import type { ILoadInfo } from '@/service/info';
import { getCurrentInfo, getUploadCount } from '@/service/info';
import { ROUTER_PATH } from '@/utils/router';
import Footer from './components/Footer';
import SocialMedia from './components/SocialMedia';
import { message } from 'antd';

const NetworkSphere = dynamic(() => import('@/components/NetworkSphere'), {
  ssr: false,
  loading: () => <div className="fixed inset-0 -z-10 w-screen h-screen overflow-hidden bg-white" />
});

export default function Home() {
  const router = useRouter();
  const [showCreate, setShowCreate] = useState(false);
  const [count, setCount] = useState<number | undefined>(undefined);
  const [isMounted, setIsMounted] = useState(false);
  const [contentVisible, setContentVisible] = useState(false);

  const [loading, setLoading] = useState(true);
  const [loadInfo, setLoadInfo] = useState<ILoadInfo | null>(null);

  useEffect(() => {
    getCurrentInfo()
      .then((res) => {
        if (res.data.code === 0) {
          setLoadInfo(res.data.data);
          localStorage.setItem('upload', JSON.stringify(res.data.data));
        }
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  useEffect(() => {
    getUploadCount()
      .then((res) => {
        if (res.data.code === 0) {
          setCount(res.data.data.count);
        } else {
          throw new Error(res.data.message);
        }
      })
      .catch((error: any) => {
        message.error(error.message || 'Failed to load upload count');
      });
  }, []);

  const handleExistingUploadClick = () => {
    router.push(ROUTER_PATH.DASHBOARD);
  };

  const handleSphereInitialized = () => {
    setTimeout(() => {
      setContentVisible(true);
    }, 300);
  };

  // Only render content on the client side
  if (!isMounted) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center p-4 relative bg-white">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-secondme-blue" />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 relative">
      {/* Network sphere background */}
      <NetworkSphere onInitialized={handleSphereInitialized} />

      {/* Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div
          className={`absolute top-20 left-20 w-64 h-64 rounded-full bg-[#4ECDC4]/10 blur-3xl delay-[400ms] transition-opacity duration-1000 ease-in-out ${contentVisible ? 'opacity-100' : 'opacity-0'}`}
        />
        <div
          className={`absolute bottom-20 right-20 w-64 h-64 rounded-full bg-[#FF6B6B]/10 blur-3xl delay-[500ms] transition-opacity duration-1000 ease-in-out ${contentVisible ? 'opacity-100' : 'opacity-0'}`}
        />
        <div
          className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 rounded-full bg-[#FFD93D]/10 blur-3xl delay-[600ms] transition-opacity duration-1000 ease-in-out ${contentVisible ? 'opacity-100' : 'opacity-0'}`}
        />
      </div>

      <div className="relative z-10 text-center mt-[-8vh] w-full overflow-visible px-4">
        <div
          className={`transition-opacity duration-700 ease-in-out ${contentVisible ? 'opacity-100' : 'opacity-0'}`}
        >
          <h1 className="text-5xl md:text-6xl font-bold mb-3 mx-auto leading-tight px-4 flex items-center justify-center">
            <img alt="Second Me Logo" className="h-20 md:h-28 mr-5" src="/images/single_logo.png" />
            <span
              className="bg-gradient-to-br from-[#1E293B] to-[#475569] bg-clip-text text-transparent drop-shadow-sm inline-block tracking-[0.01em] font-[Calistoga]"
              style={{
                textShadow: '0 2px 4px rgba(0,0,0,0.08)'
              }}
            >
              Create Your AI self
            </span>
          </h1>
          <p className="text-2xl md:text-3xl mb-14 mx-auto  px-4 flex flex-wrap justify-center tracking-[0.01em] font-[Calistoga]">
            <span className="inline-block mx-2 bg-gradient-to-br from-[#334155] to-[#475569] bg-clip-text text-transparent">
              Locally Trained
            </span>
            <span className="inline-block text-[#64748B] mx-2">·</span>
            <span className="inline-block mx-2 bg-gradient-to-br from-[#334155] to-[#475569] bg-clip-text text-transparent">
              Globally Connected
            </span>
          </p>

          <div
            className={`text-sm mb-12 transition-opacity duration-700 ease-in-out ${contentVisible ? 'opacity-100' : 'opacity-0'}`}
            style={{ transitionDelay: '400ms', color: '#64748B' }}
          >
            <span className="font-medium text-[#334155]">{count}</span>{' '}
            <span>Second Me in network</span>
          </div>
        </div>

        {!loading && (
          <div
            className={`transition-opacity duration-700 ease-in-out delay-[300ms] ${contentVisible ? 'opacity-100' : 'opacity-0'}`}
          >
            {loadInfo ? (
              <button className="btn-primary" onClick={handleExistingUploadClick}>
                Continue as {loadInfo.name}
              </button>
            ) : (
              <button className="btn-primary" onClick={() => setShowCreate(true)}>
                Create my Second Me
              </button>
            )}
          </div>
        )}
      </div>

      {showCreate && <CreateSecondMe onClose={() => setShowCreate(false)} />}

      {/* Quick examples section - Moved to the bottom of the page */}
      <Footer className={contentVisible ? 'opacity-100' : 'opacity-0'} />

      {/* Social Media Links - Fixed to bottom right */}
      <SocialMedia className={contentVisible ? 'opacity-100' : 'opacity-0'} />
    </div>
  );
}
