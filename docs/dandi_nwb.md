# DANDI / NWB access for Studies

Shared helpers for any Study whose source is **NWB on the DANDI Archive**. They live in [`src/neuralshed/datasets/nwb.py`](../src/neuralshed/datasets/nwb.py) and do two jobs: **resolve** DANDI assets (list them, build a streaming URL) and **open** an NWB/HDF5 file **local-or-remote** behind one API. A Study reads every session through them and works whether the files are on disk or streamed in place.

---

## When to stream vs. download

**Downloaded local files are the default** for most studies — the Study's `_download()` lands them under `self.path` and reads go straight to disk. **Streaming** (byte-range reads over the network, no download) is an option **for DANDI/NWB datasets only**; reach for it to explore a dataset before committing disk, or to avoid bulk-downloading a very large dandiset, which may come with heavy data that won't be useful for the decoding models (e.g. raw electrophysiology data). The same code path serves both: a local file always wins; otherwise the asset is streamed.

---

## Transport (local-or-remote)

Every opener takes a `source` (a local path **or** an `http(s)://` URL) and a `stream` switch, and returns an open `pynwb.NWBFile` or `h5py.File`.

| Function | Form | Returns | Use |
|---|---|---|---|
| `open_nwb(source, stream="auto")` | context manager | `pynwb.NWBFile` | Scoped read of the parsed NWB (loops over many sessions). Closes all handles on exit. |
| `open_h5(source, stream="auto")` | context manager | `h5py.File` | Scoped read by raw HDF5 key (what a `Spikes` extractor wants — no pynwb parse). |
| `load_nwb(source, stream="auto")` | live | `pynwb.NWBFile` | Notebooks: returns the object directly, handles stashed on `nwb._neuralshed_handles` so it stays open across cells. |
| `load_h5(source, stream="auto")` | live | `h5py.File` | A reader that **owns the handle's lifetime** (e.g. a `SpecialLoader` whose `h5py.File` the extractor closes after binning). Backing `remfile` is kept alive on the handle. |

**The `stream` switch:**
- `"auto"` (default) — stream a remote URL, read a local path directly. **A local file always wins** (a path that exists on disk is never streamed).
- `True` — force streaming.
- `False` — force a local read.

```python
from neuralshed.datasets import nwb

# scoped read (local path or DANDI URL — same call)
with nwb.open_nwb(source) as nwbfile:
    kin = nwbfile.acquisition["OpenLoopKinematics"].data[:]

# notebook: stays open across cells
nwbfile = nwb.load_nwb(source)
```

---

## DANDI resolver

Built on the official `dandi` client (`dandi.dandiapi.DandiAPIClient`). Listing and URL-building fetch **metadata only**, no NWB bytes.

| Function | Returns | Notes |
|---|---|---|
| `dandi_list_assets(dandiset, version="draft", glob=None)` | `[{"path", "asset_id", "size"}, …]` sorted by path | Optional `glob` restricts to matching paths. |
| `dandi_asset_url(dandiset, asset_path_or_id, version="draft")` | streaming download URL | Accepts a dandiset-relative path **or** a raw asset UUID. The URL redirects to S3 and supports byte-range reads. |
| `resolve_source(ref, dandiset, version="draft")` | source string | **Local-or-remote** resolver: a local file path or `http(s)` URL passes through, anything else is treated as a DANDI asset path/UUID and resolved to its streaming URL. Feed the result to the openers. |
| `open_dandi_nwb(dandiset, asset_path_or_id, version="draft", stream="auto")` | context-manager `NWBFile` | Convenience = `dandi_asset_url` + `open_nwb`. |

```python
assets = nwb.dandi_list_assets("000954")          # 40 assets, metadata only
url = nwb.dandi_asset_url("000954", assets[0]["path"])
with nwb.open_dandi_nwb("000954", assets[0]["path"]) as nwbfile:
    ...
```

---

## How streaming works

Remote reads use **`remfile` + `h5py` + `pynwb`**: `remfile.File(url)` exposes the S3 object as a seekable file with a small chunk cache, `h5py` opens the HDF5 container over it, and `pynwb` parses the NWB structure. Only the requested byte ranges are actually touched cross the wire, there is no full download.

---

## Using it from a Study (fan_h1 pattern)

A DANDI/NWB Study wires four small methods to the shared helpers. The key idea: the timeline **`path` is a relative asset path that is identical whether the file is local or streamed**, so it doubles as the cache key (see below).

```python
_DANDISET_ID = "000954"
_VERSION = "draft"

def _download(self) -> None:
    from neuralfetch.download import Dandi
    # folder="" makes the local layout mirror the dandiset: self.path/000954/sub-*/…nwb
    Dandi(study=_DANDISET_ID, version=_VERSION, dset_dir=self.path, folder="").download()

def _nwb_files_paths(self) -> list[str]:
    # local files when present, else enumerate the dandiset and stream.
    local = [p for p in self.path.rglob("*.nwb") if not p.name.startswith("._")]
    rels = (
        [str(p.relative_to(self.path)) for p in local]
        if local
        else [_DANDISET_ID + "/" + a["path"] for a in nwb.dandi_list_assets(_DANDISET_ID, _VERSION)]
    )
    return sorted(rels)

def _resolve_source(self, timeline_path: str) -> str:
    # local path when the file exists, else the DANDI streaming URL.
    local = self.path / timeline_path
    if local.is_file():
        return str(local)
    asset_path = timeline_path.removeprefix(_DANDISET_ID + "/")
    return nwb.dandi_asset_url(_DANDISET_ID, asset_path, _VERSION)

def iter_timelines(self):
    for rel in self._nwb_files_paths():
        yield dict(subject=..., session=Path(rel).stem, path=rel, ...)
```

`_load_timeline_events` then opens the resolved source to read timing/labels, and emits `SpecialLoader` rows for the bulk signal:

```python
def _load_timeline_events(self, timeline):
    with nwb.open_nwb(self._resolve_source(timeline["path"])) as nwbfile:
        ...  # read n_bins, epochs, etc.
    # Spikes row: a SpecialLoader that hands the extractor a live h5py.File
    filepath = study.SpecialLoader(method=self._load_spikes, timeline=timeline).to_json()

def _load_spikes(self, timeline):
    # extractor closes this handle after binning; load_h5 keeps remfile alive while streaming
    return nwb.load_h5(self._resolve_source(timeline["path"]))
```

Use `open_nwb` (context manager) for scoped reads inside `_load_timeline_events`. Use `load_h5`/`load_nwb` only when a downstream reader owns the handle's lifetime.

### Shared cache keys (local and streamed)

exca keys each timeline's cached events by the timeline dict, whose `path` is the **relative** asset path. Because `_download(folder="")` lays local files out as `self.path/<dandiset>/sub-*/…nwb` and `_nwb_files_paths` prefixes streamed asset paths with the same `<dandiset>/`, the `path` is **identical in both modes**. So a cache built while local is reused while streaming, and vice versa, and switching modes does not invalidate it.

---

## Requirements & gotchas

- **`requirements`:** a NWB Study should declare `requirements = ("dandi", "pynwb", "h5py", "remfile")`.
- **Handle lifetime:** `open_*` (context managers) close everything on exit. `load_*` keep the file open.
- **`"auto"` favors local.** A path that exists on disk is read directly even if you pass a URL-shaped string elsewhere. To truly force streaming of a present file, pass `stream=True`.
- **Streaming needs network**, a flaky connection surfaces as read errors deep in `h5py`. Download (`_download()`) for offline/repeated heavy use.
- **Don't bulk-download giant dandisets** just to read a few sessions, choose instead to stream them.
