'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ROUTER_PATH } from '@/utils/router';

export default function TrainRedirect() {
  const router = useRouter();

  useEffect(() => {
    router.push(ROUTER_PATH.TRAIN_IDENTITY);
  }, [router]);

  return null;
}
