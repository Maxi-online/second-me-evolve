'use client';

import { useLoadInfoStore } from '@/store/useLoadInfoStore';
import { EVENT } from '@/utils/event';
import { useEffect, useMemo } from 'react';

export default function NativeApplications() {
  const loadInfo = useLoadInfoStore((state) => state.loadInfo);
  const isRegistered = useMemo(() => {
    return loadInfo?.status === 'online';
  }, [loadInfo]);

  useEffect(() => {
    if (!isRegistered) {
      dispatchEvent(new Event(EVENT.SHOW_REGISTER_MODAL));
    }
  }, [isRegistered]);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 bg-secondme-warm-bg rounded-xl">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Second X Apps</h1>
        <p className="mt-2 text-sm text-gray-600">
          Second X Apps transforms traditional human-centric platforms into services designed for
          Second Me. These next-generation applications enable your Second Me to navigate the
          digital world autonomously, making decisions and building connections while preserving
          your time and energy.
          {/* Future services natively-built for Second Me to use: Second Tinder, Second Linkedin, etc. */}
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-2">
        {/* Second Tinder - Disabled */}
        <div className="group relative bg-white rounded-lg shadow-sm overflow-hidden ring-1 ring-black/5 opacity-50 cursor-not-allowed">
          <div className="p-6">
            <div className="flex items-center space-x-3 min-h-[24px]">
              <svg
                className="w-6 h-6 text-secondme-red"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                />
              </svg>
              <h3 className="text-lg font-medium text-gray-900">Second Tinder</h3>
              <span className="text-xs text-gray-500">(Coming Soon)</span>
            </div>
            <p className="mt-3 text-sm text-gray-500">
              A dating platform built specifically for Second Me. Your AI self can independently
              explore relationships and connections in their own social space.
            </p>
          </div>
        </div>

        {/* Second LinkedIn - Disabled */}
        <div className="group relative bg-white rounded-lg shadow-sm overflow-hidden ring-1 ring-black/5 opacity-50 cursor-not-allowed">
          <div className="p-6">
            <div className="flex items-center space-x-3 min-h-[24px]">
              <svg
                className="w-6 h-6 text-secondme-blue"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                />
              </svg>
              <h3 className="text-lg font-medium text-gray-900">Second LinkedIn</h3>
              <span className="text-xs text-gray-500">(Coming Soon)</span>
            </div>
            <p className="mt-3 text-sm text-gray-500">
              A professional networking platform designed exclusively for Second Me to build their
              own career network and professional relationships.
            </p>
          </div>
        </div>

        {/* Second Airbnb - Disabled */}
        <div className="group relative bg-white rounded-lg shadow-sm overflow-hidden ring-1 ring-black/5 opacity-50 cursor-not-allowed">
          <div className="p-6">
            <div className="flex items-center space-x-3 min-h-[24px]">
              <svg
                className="w-6 h-6 text-secondme-red"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                />
              </svg>
              <h3 className="text-lg font-medium text-gray-900">Second Airbnb</h3>
              <span className="text-xs text-gray-500">(Coming Soon)</span>
            </div>
            <p className="mt-3 text-sm text-gray-500">
              A hospitality platform where your Second Me acts as your digital property manager,
              handling guest communications and optimizing rentals.
            </p>
          </div>
        </div>

        {/* Second OnlyFans - Disabled */}
        <div className="group relative bg-white rounded-lg shadow-sm overflow-hidden ring-1 ring-black/5 opacity-50 cursor-not-allowed">
          <div className="p-6">
            <div className="flex items-center space-x-3 min-h-[24px]">
              <svg
                className="w-6 h-6 text-secondme-green"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2-1.343-2-3-2zM17 15v-2a4 4 0 00-4-4H7a4 4 0 00-4 4v2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                />
              </svg>
              <h3 className="text-lg font-medium text-gray-900">Second OnlyFans</h3>
              <span className="text-xs text-gray-500">(Coming Soon)</span>
            </div>
            <p className="mt-3 text-sm text-gray-500">
              A creator platform where your Second Me operates as a digital content creator,
              providing personalized content and engaging with subscribers.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
