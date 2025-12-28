import type { Metadata } from 'next';
import './globals.css';
import { Suspense } from 'react';
import HeaderLayout from '@/layouts/HeaderLayout';

export const metadata: Metadata = {
  title: 'Second Me',
  description: 'Train and deploy your AI self'
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html className="h-full" lang="en">
      <body className="flex flex-col font-sans antialiased h-full">
        <Suspense>
          <HeaderLayout>{children}</HeaderLayout>
        </Suspense>
      </body>
    </html>
  );
}
