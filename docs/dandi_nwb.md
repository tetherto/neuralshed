# DANDI / NWB access for Studies

Studies whose source is NWB on the DANDI Archive share the helpers in [`datasets/nwb.py`](../src/neuralshed/datasets/nwb.py). They list a dandiset's assets, build streaming URLs, and open an NWB or HDF5 file from a local path or a URL through one call. Each function's docstring describes its arguments, and [`karpowicz_2024_h1/study.py`](../src/neuralshed/datasets/karpowicz_2024_h1/study.py) is the reference implementation.

## Download or stream

Downloaded local files are the default. Streaming reads over the network only the byte ranges a call touches, and is available for DANDI/NWB sources only. Use it to explore a dandiset before committing disk space, or to avoid downloading a large dandiset whose bulk (raw electrophysiology, for example) the models do not use. A Study supports both through one code path: a file that exists locally is read from disk, any other is streamed.

## Timeline paths

The timeline `path` is the relative asset path `<dandiset>/sub-*/….nwb` in both modes. Two things make that hold:

- `_download` calls `neuralfetch.download.Dandi` with `dset_dir=self.path` and `folder=""`, so local files land at `self.path/<dandiset>/sub-*/….nwb`.
- `iter_timelines` lists the paths with `nwb.dandi_nwb_paths(dandiset, version, root=self.path)`: the local files relative to `self.path` when there are any, else the dandiset's assets prefixed with `<dandiset>/`.

When reading, `nwb.resolve_source(path, dandiset, version, root=self.path)` returns the local file if it exists, else the streaming URL. Extracted features are cached per event, keyed by the event's `filepath`, and a `SpecialLoader` filepath embeds the timeline dict. A `path` that is the same in both modes therefore keeps the extractor cache valid when switching between local and streamed reads.

## Opening files

The openers read a local path from disk and stream an `http(s)` URL.

- `nwb.open_nwb` and `nwb.open_h5` are context managers that close every handle on exit. Use them for scoped reads in `_load_timeline_events`.
- `nwb.load_h5` returns an open `h5py.File` and keeps its streaming handle alive. Use it in a `SpecialLoader` target whose consumer closes the file, such as the `Spikes` extractor.
- `nwb.load_nwb` returns an open `pynwb.NWBFile` the same way, for notebooks.

## Requirements and gotchas

- Declare `requirements = ("dandi", "pynwb", "h5py", "remfile")`.
- Streaming needs a network connection, and a flaky one surfaces as read errors deep in `h5py`. Download for offline or repeated heavy use.
