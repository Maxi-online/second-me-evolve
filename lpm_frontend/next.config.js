const nextConfig = {
  reactStrictMode: false,
  async rewrites() {
    const dockerApiBaseUrl = process.env.DOCKER_API_BASE_URL;
    const localApiBaseUrl = `${process.env.HOST_ADDRESS || 'http://127.0.0.1'}:${process.env.LOCAL_APP_PORT || 8002}`;

    return [
      {
        source: '/',
        destination: '/home'
      },
      {
        source: '/api/:path*',
        destination: dockerApiBaseUrl
          ? `${dockerApiBaseUrl}/api/:path*`
          : `${localApiBaseUrl}/api/:path*`
      }
    ];
  },
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Credentials', value: 'true' },
          { key: 'Access-Control-Allow-Origin', value: '*' },
          {
            key: 'Access-Control-Allow-Methods',
            value: 'GET,DELETE,PATCH,POST,PUT'
          },
          {
            key: 'Access-Control-Allow-Headers',
            value: 'Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date'
          },
          { key: 'Accept', value: 'text/event-stream' },
          { key: 'Cache-Control', value: 'no-cache' },
          { key: 'Connection', value: 'keep-alive' }
        ]
      }
    ];
  },
  experimental: {
    proxyTimeout: 0
  },
  compiler: {
    styledComponents: true
  },
  webpack: (config) => {
    config.externals = [...(config.externals || []), 'canvas', 'jsdom'];

    config.watchOptions = {
      poll: 1000,
      aggregateTimeout: 300
    };

    return config;
  }
};

module.exports = nextConfig;
