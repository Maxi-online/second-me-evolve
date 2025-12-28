'use client';

import DashboardLayout from '@/layouts/DashboardLayout';
import { Suspense } from 'react';

interface IProps {
  children: React.ReactNode;
}

function Layout(props: IProps): JSX.Element | React.ReactNode | null {
  const { children } = props;

  return (
    <Suspense>
      <DashboardLayout>{children}</DashboardLayout>
    </Suspense>
  );
}

export default Layout;
