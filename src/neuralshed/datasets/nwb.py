import contextlib
import os
import re
import typing as tp

# DANDI Archive REST root
DANDI_API = "https://api.dandiarchive.org/api"

_UUID_RE = re.compile(r"[0-9a-fA-F-]{36}")
Stream = tp.Union[bool, tp.Literal["auto"]]


def _is_remote(source: str) -> bool:
    return "://" in source


def _should_stream(source: str, stream: Stream) -> bool:
    """Resolve ``stream`` to a bool. ``"auto"`` streams remote URLs and reads
    local paths directly (a local file always wins over streaming)."""
    if stream == "auto":
        return _is_remote(source) and not os.path.isfile(source)
    return bool(stream)


def _open_h5(source: str, stream: Stream):
    """Return ``(h5py.File, remfile_or_None)``. ``remfile`` is ``None`` for a
    local read, else kept so the caller can close it."""
    import h5py

    if not _should_stream(source, stream):
        return h5py.File(source, "r"), None
    import remfile

    remf = remfile.File(source)
    return h5py.File(remf, "r"), remf


def _close(*handles: tp.Any) -> None:
    for h in handles:
        if h is not None:
            with contextlib.suppress(Exception):
                h.close()


@contextlib.contextmanager
def open_h5(source: str, stream: Stream = "auto") -> tp.Iterator[tp.Any]:
    """Yield an open ``h5py.File`` (local or streamed); handles closed on exit.

    This is the rung the ``Spikes`` extractor wants -- it reads ``units/...``
    by HDF5 key and never needs the pynwb object.
    """
    h5f, remf = _open_h5(str(source), stream)
    try:
        yield h5f
    finally:
        _close(h5f, remf)


@contextlib.contextmanager
def open_nwb(source: str, stream: Stream = "auto") -> tp.Iterator[tp.Any]:
    """Yield a parsed ``pynwb.NWBFile`` (local or streamed). Context manager;
    all handles (IO, h5py, remfile) are closed on exit.

    Parameters
    ----------
    source : str
        A local file path or a full ``http(s)://`` URL (e.g. from
        :func:`dandi_asset_url`).
    stream : {"auto", True, False}
        ``"auto"`` (default) streams a remote URL and reads a local path
        directly. ``False`` forces a local read; ``True`` forces streaming.
    """
    import pynwb

    h5f, remf = _open_h5(str(source), stream)
    io = pynwb.NWBHDF5IO(file=h5f, mode="r", load_namespaces=True)
    try:
        yield io.read()
    finally:
        _close(io, h5f, remf)


def load_nwb(source: str, stream: Stream = "auto") -> tp.Any:
    """Open and return a live ``pynwb.NWBFile`` for interactive/notebook use.

    Unlike :func:`open_nwb` (a context manager), the handles are stashed on the
    returned object (``nwb._neuralshed_handles``) so it stays open until garbage
    collected. For deterministic cleanup (loops over many files), prefer
    :func:`open_nwb`.
    """
    import pynwb

    h5f, remf = _open_h5(str(source), stream)
    io = pynwb.NWBHDF5IO(file=h5f, mode="r", load_namespaces=True)
    nwb = io.read()
    nwb._neuralshed_handles = (io, h5f, remf)  # type: ignore[attr-defined]
    return nwb


def load_h5(source: str, stream: Stream = "auto") -> tp.Any:
    """Open and return a live ``h5py.File`` (local or streamed) for the caller to
    close. Use when a reader needs the raw HDF5 handle and owns its lifetime --
    e.g. a ``Spikes`` ``SpecialLoader`` whose handle ``SpikesExtractor`` closes
    after binning. When streaming, the backing ``remfile`` stays alive while the
    file is open. For scoped use prefer :func:`open_h5`.
    """
    h5f, remf = _open_h5(str(source), stream)
    if remf is not None:
        h5f._neuralshed_remf = remf  # type: ignore[attr-defined]  # keep reader alive
    return h5f


# --- DANDI resolver (official dandi client) -------------------------------
def dandi_list_assets(
    dandiset: str,
    version: str = "draft",
    glob: str | None = None,
) -> list[dict[str, tp.Any]]:
    """List a dandiset's assets -- metadata only, no NWB bytes fetched.

    Parameters
    ----------
    dandiset : str
        Dandiset id, e.g. ``"000954"``.
    version : str
        Dandiset version (default ``"draft"``).
    glob : str, optional
        If given, restrict to assets whose path matches this glob.

    Returns
    -------
    list of dict
        One ``{"path", "asset_id", "size"}`` per asset, sorted by path.
    """
    from dandi.dandiapi import DandiAPIClient

    with DandiAPIClient() as client:
        ds = client.get_dandiset(dandiset, version, lazy=True)
        assets = ds.get_assets_by_glob(glob) if glob else ds.get_assets()
        out = [
            {"path": a.path, "asset_id": a.identifier, "size": a.size} for a in assets
        ]
    out.sort(key=lambda a: a["path"])
    return out


def dandi_asset_url(
    dandiset: str,
    asset_path_or_id: str,
    version: str = "draft",
) -> str:
    """Return an asset's streaming download URL (for ``remfile``/:func:`open_nwb`).

    ``asset_path_or_id`` may be a dandiset-relative path or a raw asset UUID.
    The URL 302-redirects to S3 and supports byte-range reads; it is meant for
    streaming, not a full download.
    """
    if "/" not in asset_path_or_id and _UUID_RE.fullmatch(asset_path_or_id):
        aid = asset_path_or_id
    else:
        from dandi.dandiapi import DandiAPIClient

        with DandiAPIClient() as client:
            ds = client.get_dandiset(dandiset, version, lazy=True)
            aid = ds.get_asset_by_path(asset_path_or_id).identifier
    return f"{DANDI_API}/dandisets/{dandiset}/versions/{version}/assets/{aid}/download/"


def resolve_source(ref: str, dandiset: str, version: str = "draft") -> str:
    """Resolve a reference to a source string for the openers (local-or-remote).

    A local file path or an ``http(s)://`` URL passes through unchanged; anything
    else is treated as a DANDI asset path or UUID in ``dandiset`` and resolved to
    its streaming URL via :func:`dandi_asset_url`. Feed the result to
    :func:`open_nwb` / :func:`open_h5` / :func:`load_nwb` / :func:`load_h5`.
    """
    if os.path.isfile(ref) or _is_remote(ref):
        return ref
    return dandi_asset_url(dandiset, ref, version)


@contextlib.contextmanager
def open_dandi_nwb(
    dandiset: str,
    asset_path_or_id: str,
    version: str = "draft",
    stream: Stream = "auto",
) -> tp.Iterator[tp.Any]:
    """Convenience: resolve a DANDI asset and open it as an ``NWBFile``.

    Composes :func:`dandi_asset_url` with :func:`open_nwb`. With ``stream`` at
    ``"auto"`` the asset is streamed (the resolved source is a remote URL); pass
    a local path to :func:`open_nwb` directly if the file is already on disk.
    """
    url = dandi_asset_url(dandiset, asset_path_or_id, version)
    with open_nwb(url, stream=stream) as nwb:
        yield nwb
