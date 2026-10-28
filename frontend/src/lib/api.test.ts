import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
describe("API boundary", () => {
  beforeEach(() => { vi.resetModules(); vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "https://backend.example"); });
  afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });
  it("preserves empty results", async () => { vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ results: [] })))); const { request } = await import("./api"); expect(await request("/api/results/summary")).toEqual({ results: [] }); });
  it("reports backend service errors", async () => { vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Checkpoint unavailable" }), { status: 503 }))); const { request } = await import("./api"); await expect(request("/api/predict")).rejects.toThrow("Checkpoint unavailable"); });
  it("explains offline backend", async () => { vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch"))); const { request } = await import("./api"); await expect(request("/health")).rejects.toThrow("Cannot reach"); });
  it("requires configuration", async () => { vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", ""); const { request } = await import("./api"); await expect(request("/health")).rejects.toThrow("not configured"); });
});
