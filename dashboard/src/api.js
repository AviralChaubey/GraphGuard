// Thin client over the frozen API contract (docs/CONTRACTS.md, Contract 3).
// No env-var setup needed for local dev: falls back to the FastAPI default host:port.
const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(path, options) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${options?.method || "GET"} ${path} -> ${res.status}: ${body}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),
  listRings: (limit = 20, beforeTimeStep) =>
    request(
      `/rings?limit=${limit}${
        beforeTimeStep === undefined || beforeTimeStep === null || beforeTimeStep === ""
          ? ""
          : `&before_time_step=${encodeURIComponent(beforeTimeStep)}`
      }`
    ),
  getRing: (ringId) => request(`/ring/${encodeURIComponent(ringId)}`),
  getExplanation: (targetId) => request(`/explain/${encodeURIComponent(targetId)}`),
  sendFeedback: (ringId, verdict, note) =>
    request("/feedback", {
      method: "POST",
      body: JSON.stringify({ ring_id: ringId, verdict, note }),
    }),
};
