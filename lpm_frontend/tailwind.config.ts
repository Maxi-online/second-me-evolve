import type { Config } from 'tailwindcss';

export default {
  content: [
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    './src/layouts/**/*.{js,ts,jsx,tsx,mdx}'
  ],
  theme: {
    extend: {
      colors: {
        background: 'var(--background)',
        foreground: 'var(--foreground)',
        secondme: {
          'warm-bg': '#FDF8F3',
          blue: '#4A90E2',
          green: '#50B86B',
          red: '#FF6B6B',
          yellow: '#FFD93D',
          navy: '#2C3E50',
          gray: {
            100: '#F7F9FA',
            200: '#E9ECEF',
            300: '#DEE2E6',
            400: '#CED4DA',
            500: '#ADB5BD',
            600: '#6C757D',
            700: '#495057',
            800: '#343A40',
            900: '#212529'
          }
        }
      }
    }
  },
  plugins: []
} satisfies Config;
