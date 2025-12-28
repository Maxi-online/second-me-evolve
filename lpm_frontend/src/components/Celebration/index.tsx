'use client';

import type React from 'react';
import { useState, useEffect } from 'react';
import confetti from 'canvas-confetti';
import { motion } from 'framer-motion';
import { useLoadInfoStore } from '@/store/useLoadInfoStore';

interface CelebrationEffectProps {
  isVisible: boolean;
  onClose: () => void;
}

const CelebrationEffect: React.FC<CelebrationEffectProps> = ({ isVisible, onClose }) => {
  const [showMessage, setShowMessage] = useState(false);
  const loadInfo = useLoadInfoStore((state) => state.loadInfo);
  const secondMeName = loadInfo?.name || 'Second Me';

  function randomInRange(min: number, max: number) {
    return Math.random() * (max - min) + min;
  }

  useEffect(() => {
    if (isVisible) {
      // Immediately show the message to prevent height changes
      setShowMessage(true);

      // Trigger confetti effect with more realistic settings
      const duration = 5 * 1000;
      const animationEnd = Date.now() + duration;
      const defaults = {
        startVelocity: 35,
        spread: 360,
        ticks: 100,
        zIndex: 100,
        gravity: 1.2,
        drift: 0,
        scalar: 1.2,
        colors: [
          '#5D8BF4',
          '#4CC9F0',
          '#7209B7',
          '#F72585',
          '#4361EE',
          '#FFD700',
          '#00FF00',
          '#FF4500'
        ]
      };

      const interval: any = setInterval(function () {
        const timeLeft = animationEnd - Date.now();

        if (timeLeft <= 0) {
          return clearInterval(interval);
        }

        const particleCount = 50 * (timeLeft / duration);

        // Launch colorful confetti from both sides
        confetti({
          ...defaults,
          particleCount,
          origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 }
        });
        confetti({
          ...defaults,
          particleCount,
          origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 }
        });
      }, 250);

      // No auto-close timer - only close when user clicks the button

      return () => {
        clearInterval(interval);
        setShowMessage(false);
      };
    }
  }, [isVisible, onClose]);

  // Don't render anything if not visible
  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 flex items-center justify-center z-[100]">
      <motion.div
        animate={{ opacity: 1 }}
        className="bg-secondme-warm-bg p-10 rounded-2xl shadow-2xl max-w-md w-full h-[380px] text-center border-2 border-gray-800/10 relative overflow-hidden"
        initial={{ opacity: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Background gradient decorations */}
        <div className="absolute -top-20 -right-20 w-48 h-48 rounded-full bg-orange-50 opacity-70" />
        <div className="absolute -bottom-20 -left-20 w-48 h-48 rounded-full bg-orange-50 opacity-70" />
        <div className="relative z-10 h-full flex flex-col justify-center">
          <div
            className={`transition-opacity duration-500 ${showMessage ? 'opacity-100' : 'opacity-0'}`}
          >
            <motion.div
              animate={{
                scale: [1, 1.2, 1],
                rotate: [0, 3, -3, 0],
                y: [0, -10, 0]
              }}
              className="mb-6 flex justify-center"
              transition={{
                repeat: Infinity,
                repeatType: 'reverse',
                duration: 2.5
              }}
            >
              <div className="relative">
                <span className="text-6xl filter drop-shadow-lg">✨</span>
                <motion.div
                  animate={{ opacity: [0.5, 1, 0.5], scale: [0.8, 1.1, 0.8] }}
                  className="absolute top-0 left-0 right-0 bottom-0 flex items-center justify-center"
                  transition={{ repeat: Infinity, duration: 3 }}
                >
                  <span className="text-6xl">🌟</span>
                </motion.div>
              </div>
            </motion.div>

            <motion.h2
              animate={{ y: [0, -5, 0] }}
              className="text-3xl font-bold text-gray-900 mb-3"
              transition={{ repeat: 2, duration: 1 }}
            >
              Training Complete!
            </motion.h2>
            <div>
              <p className="text-gray-700 mb-2 text-lg">
                <span className="font-bold">{secondMeName}</span> has been born
              </p>
              <p className="text-gray-600 mb-4 text-sm leading-relaxed">
                Your Second Me has learned from your identity and memories, and is now ready to
                chat, share spaces, and connect with other AIs in the network.
              </p>
            </div>

            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <button
                className="px-6 py-2.5 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors font-medium shadow-[3px_3px_0px_0px_rgba(0,0,0,0.1)]"
                onClick={onClose}
              >
                Start the journey!
              </button>
            </motion.div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default CelebrationEffect;
