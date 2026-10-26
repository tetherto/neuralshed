import type { CSSProperties } from "react";

// Target family color as a theme-aware CSS variable (--fam-text/-voice/-movement,
// which flip in dark mode), not the backend's fixed light-mode hex.
export function famVar(target: string): string {
  return `var(--fam-${target}, var(--ink-faint))`;
}

// Inline style carrying the --fam custom property for a card/dot.
export function famStyle(target: string): CSSProperties {
  return { ["--fam"]: famVar(target) } as CSSProperties;
}

// Collection names are identifiers (URLs, filenames, cache uids). Mirrors COLLECTION_NAME_RE
// in database/schema.py, which is what actually enforces it.
export const COLLECTION_NAME_RE = /^[A-Za-z0-9_-]+$/;

export function isValidCollectionName(name: string): boolean {
  return COLLECTION_NAME_RE.test(name.trim());
}

export function fmtChannels(channels: number[]): string {
  return channels.length ? channels.join("/") : "?";
}

export function fmtRates(rates: number[]): string {
  return rates.length ? rates.map((r) => `${r}`).join("/") : "?";
}
