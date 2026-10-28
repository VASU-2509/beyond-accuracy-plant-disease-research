"use client";
import { useEffect, useState, useCallback } from "react";
import { request } from "@/lib/api";
export function useApi<T>(path: string) {
  const [data, setData] = useState<T | null>(null), [error, setError] = useState<string | null>(null), [loading, setLoading] = useState(true), [revision, setRevision] = useState(0);
  const retry = useCallback(() => setRevision(v => v + 1), []);
  useEffect(() => {
    const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 30000); let active = true;
    setLoading(true); setError(null);
    request<T>(path, { signal: controller.signal }).then(value => { if (active) setData(value); }).catch((e: Error) => { if (active) setError(e.name === "AbortError" ? "The backend took too long to respond. Try again." : e.message); }).finally(() => { clearTimeout(timeout); if (active) setLoading(false); });
    return () => { active = false; clearTimeout(timeout); controller.abort(); };
  }, [path, revision]);
  return { data, error, loading, retry };
}
