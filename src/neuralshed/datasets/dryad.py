"""Shared Dryad downloader for neuralshed datasets.

Extends ``neuralfetch.download.Dryad`` with:

* **file selection** -- ``include`` / ``exclude`` glob patterns on the Dryad file
  path (whitelist + blacklist), so a study fetches only its in-scope files;
* **size gating** -- ``max_size`` skips files above a byte ceiling (the multi-GB
  archives some deposits ship);
* **tar extraction** -- ``.tar`` / ``.tar.gz`` / ``.tgz`` archives (upstream only
  handles ``.zip``).

Dryad gates all file downloads behind an API token, pass ``token=`` or export
``NEURALFETCH_DRYAD_TOKEN``. Listing/:meth:`preview` still works without one, as does
:func:`latest_files`, which returns each file's size and sha-256 for verifying a tree.
"""

import fnmatch
import os
import tarfile
import typing as tp
from pathlib import Path

from neuralfetch.download import Dryad as _Dryad
from neuralfetch.download import download_file, dryad_msg, extract_zip

_TAR_SUFFIXES = (".tar.gz", ".tgz", ".tar")
_ARCHIVE_SUFFIXES = _TAR_SUFFIXES + (".zip",)


def _archive_stem(name: str) -> str:
    """Filename minus its archive suffix (``sentences.tar.gz`` -> ``sentences``)."""
    for suffix in _ARCHIVE_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


class Dryad(_Dryad):
    """Public-friendly Dryad downloader with file selection (see module docstring).

    Parameters (beyond ``study`` / ``dset_dir`` / ``folder`` / ``doi``)
    ----------
    include : list[str]
        Glob whitelist on the Dryad file path; empty = every file.
    exclude : list[str]
        Glob blacklist on the Dryad file path (applied after ``include``).
    max_size : int | None
        Skip files larger than this many bytes (e.g. a multi-GB archive).

    Call :meth:`preview` to see the selected files without downloading.
    """

    token: str | None = None  # env fallback; needed to download (not to preview)
    include: list[str] = []
    exclude: list[str] = []
    max_size: int | None = None

    def model_post_init(self, log__: tp.Any) -> None:
        # Replicate BaseDownload's dir setup but drop the upstream token requirement.
        super(_Dryad, self).model_post_init(log__)
        if not self.token:
            self.token = os.environ.get("NEURALFETCH_DRYAD_TOKEN")  # optional

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _api_get(self, path: str) -> dict[str, tp.Any]:
        """Authenticated API GET with a practical timeout and transient retries."""
        import requests

        for attempt in range(1, 4):
            try:
                response = requests.get(
                    self._dryad_url(path),
                    headers=self._auth_headers(),
                    timeout=120,
                )
                if response.status_code == 401 and self.token:
                    raise RuntimeError(
                        "Dryad rejected the API token (expired or revoked). "
                        "Generate a new one and update NEURALFETCH_DRYAD_TOKEN.\n"
                        + dryad_msg
                    )
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                if attempt == 3:
                    raise
                print(f"Dryad API request failed, retrying {attempt}/2")
        raise RuntimeError("unreachable")

    def _latest_version_files(self) -> list[dict[str, tp.Any]]:
        """All files in the latest version, following Dryad's pagination.

        ``neuralfetch`` currently reads only the first API page (20 files), which can
        silently mark larger deposits as successfully downloaded while omitting the rest.
        """
        from urllib.parse import quote

        encoded = quote(f"doi:{self.doi}", safe="")
        versions = self._api_get(f"/datasets/{encoded}/versions")
        all_versions = versions.get("_embedded", {}).get("stash:versions", [])
        if not all_versions:
            raise RuntimeError(f"No versions found for DOI {self.doi}")

        href = all_versions[-1]["_links"]["stash:files"]["href"]
        files: list[dict[str, tp.Any]] = []
        while href:
            response = self._api_get(href)
            files.extend(response.get("_embedded", {}).get("stash:files", []))
            href = response.get("_links", {}).get("next", {}).get("href", "")
        return files

    def _selected(self, name: str) -> bool:
        if self.include and not any(fnmatch.fnmatch(name, p) for p in self.include):
            return False
        return not any(fnmatch.fnmatch(name, p) for p in self.exclude)

    def _select_files(self) -> list[dict[str, tp.Any]]:
        """Latest-version files passing the include/exclude/size filters."""
        files = self._latest_version_files()
        chosen = [f for f in files if self._selected(f.get("path", ""))]
        if self.max_size is not None:
            chosen = [f for f in chosen if int(f.get("size", 0) or 0) <= self.max_size]
        skipped = len(files) - len(chosen)
        print(f"Dryad {self.doi}: {len(chosen)}/{len(files)} files selected", end="")
        print(f" ({skipped} filtered out)." if skipped else ".")
        return chosen

    def preview(self) -> list[str]:
        """List the files the current filters select, without downloading.

        Does not touch the success file, so a later real run still proceeds.
        """
        chosen = self._select_files()
        for f in chosen:
            size = int(f.get("size", 0) or 0)
            print(f"  would fetch {f.get('path')} ({size / 1e6:.1f} MB)")
        return [f.get("path", "") for f in chosen]

    def _download(self, overwrite: bool = False) -> None:
        # Always per-file: a filtered/size-gated subset can't come from the bulk zip.
        self._download_individual_files(overwrite=overwrite)

    def _download_individual_files(self, overwrite: bool = False) -> None:
        chosen = self._select_files()
        if not chosen:
            raise RuntimeError(
                f"Dryad {self.doi}: no files matched the configured filters; "
                "refusing to write a successful-download marker."
            )
        for i, f in enumerate(chosen, 1):
            self._fetch_one(i, len(chosen), f, overwrite=overwrite)

    def _fetch_one(
        self,
        i: int,
        n: int,
        f: dict[str, tp.Any],
        overwrite: bool = False,
    ) -> None:
        name = f["path"]
        is_tar = name.lower().endswith(_TAR_SUFFIXES)
        is_archive = is_tar or name.lower().endswith(".zip")
        dest = self._dl_dir / name
        extracted = self._dl_dir / _archive_stem(name) if is_archive else None

        # Already-extracted output present -> done. A plain file on disk -> done.
        # An archive still on disk is *not* done: fall through and extract it.
        # ``overwrite`` forces a fresh transfer regardless of what is on disk.
        if not overwrite:
            if extracted is not None and extracted.exists():
                print(f"  [{i}/{n}] {name} already extracted, skipping")
                return
            if not is_archive and dest.exists():
                print(f"  [{i}/{n}] {name} already present, skipping")
                return

        if (
            overwrite or not dest.exists()
        ):  # download unless resuming an interrupted run
            dl_href = f.get("_links", {}).get("stash:download", {}).get("href")
            if not dl_href:
                print(f"  [{i}/{n}] {name}: no download link, skipping")
                return
            digest_type = str(f.get("digestType", "")).replace("-", "")
            digest = f.get("digest", "")
            file_hash = f"{digest_type}:{digest}" if digest and digest_type else None
            print(f"  [{i}/{n}] downloading {name}...")
            for attempt in range(1, 4):
                try:
                    download_file(
                        self._dryad_url(dl_href),
                        dest,
                        headers=self._auth_headers(),
                        show_progress=True,
                        file_hash=file_hash,
                    )
                    break
                except Exception as e:
                    if any(c in str(e) for c in ("401", "403")) and not self.token:
                        raise RuntimeError(
                            "Dryad denied the download (token required).\n" + dryad_msg
                        ) from e
                    if attempt == 3:
                        raise
                    print(
                        f"    download failed ({type(e).__name__}), retrying {attempt}/2"
                    )

        if is_tar:  # always unpack a present archive
            print(f"  [{i}/{n}] extracting {name}...")
            self._extract_tar(dest)
        elif is_archive:
            print(f"  [{i}/{n}] extracting {name}...")
            extract_zip(dest, self._dl_dir, remove_after=True)

    def _extract_tar(self, archive: Path) -> None:
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(self._dl_dir)
        archive.unlink(missing_ok=True)


def latest_files(doi: str) -> dict[str, tuple[int, str]]:
    """``path -> (size, sha-256)`` of a deposit's latest version, from the public API."""
    from urllib.parse import quote

    import requests

    api = "https://datadryad.org/api/v2"
    versions = requests.get(
        f"{api}/datasets/{quote(f'doi:{doi}', safe='')}/versions", timeout=60
    )
    versions.raise_for_status()
    href = versions.json()["_embedded"]["stash:versions"][-1]["_links"]["stash:files"][
        "href"
    ]
    out = {}
    while href:
        page = requests.get(f"https://datadryad.org{href}", timeout=60)
        page.raise_for_status()
        body = page.json()
        for f in body["_embedded"]["stash:files"]:
            out[f["path"]] = (int(f.get("size") or 0), str(f.get("digest") or ""))
        href = body.get("_links", {}).get("next", {}).get("href", "")
    return out
