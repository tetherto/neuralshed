import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { getReadme } from "../api";

type Props = {
  datasetId: string | null;
  onClose: () => void;
};

// http(s) links open in a new tab; relative links (../other/README.md, utils.py) render as
// plain text so they don't navigate the app to a dead URL.
const components: Components = {
  a({ href, children }) {
    if (href && /^https?:\/\//.test(href)) {
      return (
        <a href={href} target="_blank" rel="noreferrer">
          {children}
        </a>
      );
    }
    return <span>{children}</span>;
  },
};

export function ReadmeDrawer({ datasetId, onClose }: Props) {
  const [md, setMd] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const open = datasetId !== null;

  useEffect(() => {
    if (!datasetId) return;
    setMd(null);
    setErr(null);
    getReadme(datasetId)
      .then((r) => setMd(r.markdown))
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, [datasetId]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  return (
    <>
      <div className={`drawer-backdrop${open ? " open" : ""}`} onClick={onClose} />
      <aside className={`drawer${open ? " open" : ""}`} aria-hidden={!open}>
        <div className="drawer-head">
          <span className="drawer-id">{datasetId ?? ""}</span>
          <button className="drawer-close" type="button" onClick={onClose} aria-label="close README">
            ×
          </button>
        </div>
        <div className="drawer-body readme-body">
          {err && <p className="status-line error">Could not load README: {err}</p>}
          {!err && md === null && open && <p className="status-line">Loading README…</p>}
          {md !== null && (
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
              {md}
            </ReactMarkdown>
          )}
        </div>
      </aside>
    </>
  );
}
