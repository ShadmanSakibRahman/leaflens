/** @type {import('next').NextConfig} */
// NEXT_PUBLIC_BASE_PATH is "/leaflens" for GitHub Pages, "" for local.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";

const nextConfig = {
  reactStrictMode: true,
  output: "export",
  images: { unoptimized: true },
  basePath: basePath || undefined,
  trailingSlash: true,
  webpack: (config) => {
    // onnxruntime-web references node built-ins for its node backend; stub them
    // out for the browser build.
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
    };
    return config;
  },
};

module.exports = nextConfig;
