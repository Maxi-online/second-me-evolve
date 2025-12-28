'use client';

import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import OnboardingTutorial from '@/components/OnboardingTutorial';
import { useTrainingStore } from '@/store/useTrainingStore';
import Menu from './Menu';
import { ROUTER_PATH } from '@/utils/router';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();

  const checkTrainStatus = useTrainingStore((state) => state.checkTrainStatus);

  const [showTutorial, setShowTutorial] = useState(false);

  // Initialize training status check
  useEffect(() => {
    checkTrainStatus();
  }, []);

  return (
    <div className="h-full flex">
      {/* Fixed sidebar */}
      <Menu />

      {/* Scrollable content area */}
      {/* Height minus Menu */}
      <div className="flex-1 h-[calc(100vh-64px)] overflow-scroll no-scrollbar">
        {showTutorial ? null : children}
      </div>

      {/* OnboardingTutorial */}
      {showTutorial && (
        <OnboardingTutorial
          onClose={() => setShowTutorial(false)}
          onComplete={() => {
            setShowTutorial(false);
            router.push(ROUTER_PATH.TRAIN_IDENTITY);
          }}
        />
      )}
    </div>
  );
}
