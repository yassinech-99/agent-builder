import { initApiPassthrough } from "langgraph-nextjs-api-passthrough";

// Same-origin proxy from `/api/*` to the LangGraph agent runtime server (`http://127.0.0.1:2024`).
export const { GET, POST, PUT, PATCH, DELETE, OPTIONS, runtime } =
  initApiPassthrough({
    apiUrl: process.env.LANGGRAPH_API_URL || "http://127.0.0.1:2024",
    apiKey: process.env.LANGSMITH_API_KEY || undefined,
    runtime: "edge",
  });
