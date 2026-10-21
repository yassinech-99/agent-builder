/** @type {import('next').NextConfig} */
const nextConfig = {
  allowedDevOrigins: ["*.e2b.app", "*.e2b.dev", "localhost", "127.0.0.1"],
  experimental: {
    serverActions: {
      bodySizeLimit: "25mb",
    },
  },
  async rewrites() {
    const builderInternal =
      process.env.BUILDER_API_INTERNAL_URL || "http://127.0.0.1:8100";
    return {
      beforeFiles: [
        {
          source: "/api/builder/:path*",
          destination: `${builderInternal.replace(/\/$/, "")}/:path*`,
        },
      ],
    };
  },
};

export default nextConfig;
