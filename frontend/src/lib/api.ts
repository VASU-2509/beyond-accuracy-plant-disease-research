import type { Prediction } from "./types";
export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");
export async function request<T>(path: string, options?: RequestInit): Promise<T> {
  if (!API_BASE) throw new Error("Backend URL is not configured. Set NEXT_PUBLIC_API_BASE_URL and restart the frontend.");
  let response: Response;
  try { response = await fetch(`${API_BASE}${path}`, { ...options, signal: options?.signal ?? AbortSignal.timeout(60000) }); }
  catch (error) { if (error instanceof Error && error.name === "AbortError") throw error; throw new Error("Cannot reach the research backend. Start FastAPI or check the configured backend URL."); }
  if (!response.ok) { const payload = await response.json().catch(() => null); throw new Error(typeof payload?.detail === "string" ? payload.detail : `Request failed (${response.status}). Please check your input and try again.`); }
  return response.json() as Promise<T>;
}
export function predict(file: File, modelId: string) { const data = new FormData(); data.append("file", file); data.append("model_id", modelId); return request<Prediction>("/api/predict", { method: "POST", body: data }); }
