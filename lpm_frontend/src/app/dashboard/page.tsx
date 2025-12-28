'use client';

import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import { ROUTER_PATH } from '@/utils/router';

export default function DashboardPage() {
  const router = useRouter();

  useEffect(() => {
    router.push(ROUTER_PATH.TRAIN_IDENTITY);
  }, [router]);

  return null;
}
