'use client';

import { ROUTER_PATH } from '@/utils/router';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function PlaygroundPage() {
  const router = useRouter();

  useEffect(() => {
    router.replace(ROUTER_PATH.PLAYGROUND_CHAT);
  }, []);

  return null;
}
