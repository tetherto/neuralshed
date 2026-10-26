import type { Analysis, Bootstrap, SavedCollection, StreamReport } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export function getBootstrap(): Promise<Bootstrap> {
  return fetch("/api/bootstrap").then((r) => json<Bootstrap>(r));
}

export function analyze(ids: string[]): Promise<Analysis> {
  return fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  }).then((r) => json<Analysis>(r));
}

export function checkStreams(
  ids: string[],
  inputs: string[],
  targets: string[],
): Promise<StreamReport> {
  return fetch("/api/streams", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ids,
      input_event_types: inputs,
      target_event_types: targets,
    }),
  }).then((r) => json<StreamReport>(r));
}

export function saveCollection(
  name: string,
  ids: string[],
  chain: Record<string, unknown> | null,
  inputs: string[],
  targets: string[],
): Promise<{ saved: string; collections: SavedCollection[] }> {
  return fetch("/api/collections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      ids,
      chain,
      input_event_types: inputs,
      target_event_types: targets,
    }),
  }).then((r) => json(r));
}

export function deleteCollection(
  name: string,
): Promise<{ deleted: boolean; collections: SavedCollection[] }> {
  return fetch(`/api/collections/${encodeURIComponent(name)}`, { method: "DELETE" }).then((r) =>
    json(r),
  );
}

export function getReadme(id: string): Promise<{ id: string; markdown: string }> {
  return fetch(`/api/datasets/${encodeURIComponent(id)}/readme`).then((r) =>
    json<{ id: string; markdown: string }>(r),
  );
}
