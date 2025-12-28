'use client';

import { useEffect, useState } from 'react';
import { GithubOutlined } from '@ant-design/icons';

export default function GitHubStars() {
  const [stars, setStars] = useState<number | null>(null);

  useEffect(() => {
    fetch('https://api.github.com/repos/mindverse/Second-Me')
      .then((res) => res.json())
      .then((data) => {
        if (data.stargazers_count !== undefined) {
          setStars(data.stargazers_count);
        }
      })
      .catch((err) => {
        console.error('Failed to fetch stars', err);
      });
  }, []);

  const formatNumber = (_stars: number): string => {
    return _stars >= 1000 ? (_stars / 1000).toFixed(1) + 'k' : _stars.toString();
  };

  return (
    <div className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded transition text-gray-800">
      <GithubOutlined className="text-lg" />
      {stars !== null ? formatNumber(stars) : 'Star On GitHub'}
    </div>
  );
}
