'use client';

export default function StandaloneLayout({ children }: { children: React.ReactNode }) {
  return <div className="flex items-center justify-center bg-gray-50 h-full">{children}</div>;
}
