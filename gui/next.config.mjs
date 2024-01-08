/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    appDir: true,
  },
  images: {
    domains: ["api.rawg.io", "media.rawg.io"],
  },
  rewrites: async () => {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.API_URL}/:path*`,
        // destination: "http://127.0.0.1:5000/:path",
      },
    ]
  },
}

export default nextConfig
