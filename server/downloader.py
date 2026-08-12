"""In-container model download manager (v2.2 Step 4).

Runs the `hf download` steps in Python so the wizard can pull the selected
backends' weights on a background thread and stream per-asset progress to the
browser. Online only during setup; the runtime stays `HF_HUB_OFFLINE=1`.

Asset kinds:
    - ``snapshot`` — an HF repo (optionally pattern-limited) into the hub cache
  - ``file``     — a single file from an HF repo (the GGUF SLM)
  - ``url``      — a plain HTTP file (json.gbnf grammar)

The heavy `huggingface_hub` / network calls are injected (defaulting to lazy
real implementations) so tests drive the manager with fakes — no network.
"""

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from core.backends import DOMAINS

logger = logging.getLogger(__name__)

GRAMMAR_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/grammars/json.gbnf"
BGE_REPO = "BAAI/bge-small-en-v1.5"

# Written into a dir_snapshot dest on successful completion. Lets us
# distinguish a complete download from an interrupted one whose directory
# already exists but is only partially populated.
COMPLETE_SENTINEL = ".fulloch_complete"

# Asset states.
PENDING = "pending"
DOWNLOADING = "downloading"
DONE = "done"
ERROR = "error"


def _hf_access_denied(error: Exception) -> bool:
    """Whether a Hugging Face failure is likely resolved by an access token."""
    message = f"{type(error).__name__}: {error}".lower()
    return any(marker in message for marker in (
        "gatedrepo",
        "gated repo",
        "access to model",
        "access denied",
        "authentication",
        "unauthorized",
        "forbidden",
        "401 client error",
        "403 client error",
        "status code 401",
        "status code 403",
    ))


@dataclass
class Asset:
    key: str
    label: str
    kind: str  # "snapshot" | "file" | "url" | "dir_snapshot"
    dest: str  # directory (snapshot/file) or file path (url)
    repo: Optional[str] = None
    filename: Optional[str] = None
    url: Optional[str] = None
    allow: Optional[list] = None  # optional snapshot_download allow_patterns
    revision: Optional[str] = None
    size_gb: Optional[float] = None
    status: str = PENDING
    error: Optional[str] = None
    needs_hf_token: bool = False
    bytes_done: int = field(default=0, compare=False, repr=False)
    bytes_total: Optional[int] = field(default=None, compare=False, repr=False)

    def snapshot(self) -> dict:
        if self.status == DONE:
            pct = 100
        elif self.status == DOWNLOADING and self.bytes_total:
            pct = min(99, round(self.bytes_done / self.bytes_total * 100))
        elif self.status == PENDING:
            pct = 0
        else:
            pct = None  # indeterminate — show spinner in UI
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "size_gb": self.size_gb,
            "status": self.status,
            "error": self.error,
            "needs_hf_token": self.needs_hf_token,
            "pct": pct,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            # `domain` (asr/tts/llm) lets the wizard attach a custom-path input to
            # the right model; `present` drives the install pre-scan (on disk vs
            # needs download); `dest` is the path it'll load from / download to.
            "domain": self.key.split(":")[0] if ":" in self.key else None,
            "present": _already_present(self),
            "dest": self.dest,
        }


_WEIGHT_SUFFIXES = (".gguf", ".bin", ".safetensors")


def _is_dir_model(model) -> bool:
    """A local *directory* model path (vs an HF repo id or a single weight file)."""
    if not model:
        return False
    s = str(model)
    looks_local = s.startswith(("./", "../", "/", ".\\")) or os.path.isabs(s)
    return looks_local and Path(s).suffix not in _WEIGHT_SUFFIXES


def _is_file_model(model) -> bool:
    """A single local weight file (e.g. a custom .gguf path the user points at)."""
    return bool(model) and Path(str(model)).suffix in _WEIGHT_SUFFIXES


def plan_assets(resolved: dict, models_dir: str = "./data/models") -> list:
    """Build the asset list for a resolved `models` selection + always-required.

    Always includes the BGE embedding model (semantic note search) and the
    json.gbnf grammar, regardless of the selected backends.
    """
    hub = str(Path(models_dir) / "hub")
    grammars = str(Path(models_dir) / "grammars")
    assets: list = []

    for domain in DOMAINS:
        cfg = resolved[domain]
        spec = cfg["spec"]
        model = cfg["model"]
        # Nothing to fetch for the no-LLM bypass or a remote endpoint (no HF
        # repo/file), so skip those domains.
        if not (spec.hf_repo or spec.hf_file or spec.hf_files or spec.hf_snapshots):
            continue
        if spec.hf_snapshots:
            for index, (repo, allow, revision) in enumerate(spec.hf_snapshots):
                assets.append(
                    Asset(
                        key=f"{domain}:{spec.backend}"
                        if index == 0
                        else f"{domain}:{spec.backend}:{index}",
                        label=spec.display_name
                        if index == 0
                        else f"{spec.display_name} support files",
                        kind="snapshot",
                        dest=hub,
                        repo=repo,
                        allow=list(allow),
                        revision=revision,
                        size_gb=spec.download_size_gb if index == 0 else None,
                    )
                )
            continue
        if spec.hf_files:
            # Compound directory models such as Qwen3-TTS need weights from
            # multiple HF repos (talker plus codec) in one loader directory.
            dest = str(model)
            for index, (repo, filename) in enumerate(spec.hf_files):
                assets.append(
                    Asset(
                        key=f"{domain}:{spec.backend}"
                        if index == 0
                        else f"{domain}:{spec.backend}:codec",
                        label=spec.display_name if index == 0 else f"{spec.display_name} codec",
                        kind="file",
                        dest=dest,
                        repo=repo,
                        filename=filename,
                        size_gb=spec.download_size_gb if index == 0 else None,
                    )
                )
            continue
        # A custom LOCAL model path the user points at is honoured as-is —
        # present-checked + loaded in place, regardless of the backend's default
        # form — so the scan reflects the *actual* path, not the default repo's
        # hub presence. Checked before the hf_file/snapshot defaults.
        if _is_file_model(model):
            # A single weight file: a GGUF backend or a custom .gguf path.
            fpath = Path(str(model))
            assets.append(
                Asset(
                    key=f"{domain}:{spec.backend}",
                    label=spec.display_name,
                    kind="file",
                    dest=str(fpath.parent),
                    repo=spec.hf_repo,
                    filename=fpath.name,
                    size_gb=spec.download_size_gb,
                )
            )
        elif _is_dir_model(model):
            # A directory model (the ONNX bundle, or a custom folder): snapshot
            # flat into that dir, not the hub cache, so the loader finds it.
            # `hf_allow` (if set) fetches only the needed files from a big repo.
            assets.append(
                Asset(
                    key=f"{domain}:{spec.backend}",
                    label=spec.display_name,
                    kind="dir_snapshot",
                    dest=str(model),
                    repo=spec.hf_repo,
                    allow=list(spec.hf_allow) or None,
                    size_gb=spec.download_size_gb,
                )
            )
        elif spec.hf_file:
            # hf_file backend whose model isn't a local file path -> default file.
            assets.append(
                Asset(
                    key=f"{domain}:{spec.backend}",
                    label=spec.display_name,
                    kind="file",
                    dest=models_dir,
                    repo=spec.hf_repo,
                    filename=spec.hf_file,
                    size_gb=spec.download_size_gb,
                )
            )
        else:
            assets.append(
                Asset(
                    key=f"{domain}:{spec.backend}",
                    label=spec.display_name,
                    kind="snapshot",
                    dest=hub,
                    repo=(spec.hf_repo or model),
                    size_gb=spec.download_size_gb,
                )
            )

    assets.append(
        Asset(
            key="bge",
            label="BGE-small (semantic note search)",
            kind="snapshot",
            dest=hub,
            repo=BGE_REPO,
            size_gb=0.13,
        )
    )
    assets.append(
        Asset(
            key="grammar",
            label="JSON grammar (json.gbnf)",
            kind="url",
            dest=str(Path(grammars) / "json.gbnf"),
            url=GRAMMAR_URL,
        )
    )
    return assets


# --- default (real) download implementations (lazy heavy imports) -----------


def _make_asset_tqdm(asset: Asset, lock: threading.Lock):
    """Return a silent tqdm-compatible class for the *outer* per-repo bar
    passed as `snapshot_download`'s `tqdm_class=` kwarg.

    This bar only ever sees a `total` of *files to fetch*, not bytes — HF's own
    docstring says "the tqdm_class is not passed to each individual download" —
    so it's just used to suppress console spam; real byte counts come from
    `_byte_progress` below. Must still actually iterate the wrapped iterable
    (a `concurrent.futures.Executor.map()` result iterator): swallowing it
    would silently discard exceptions raised by failed per-file downloads.

    huggingface_hub's parallel snapshot download drives per-file bars through
    ``tqdm.contrib.concurrent.ensure_lock``, which reads/writes/deletes a real
    ``_lock`` *class* attribute (not just via get_lock/set_lock calls) — it does
    `getattr(cls, '_lock', None)` and `del cls._lock`. Mirror real tqdm's
    get_lock/set_lock (which assign `cls._lock` directly) rather than stashing
    the lock in a closure var, or `del cls._lock` raises AttributeError.
    """

    class _T:
        def __init__(self, iterable=None, *, total=None, **_kw):
            self._iterable = iterable

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def __iter__(self):
            for item in self._iterable or ():
                yield item

        def update(self, n=1):
            pass

        def set_postfix(self, **_):
            pass

        def set_description(self, *_a, **_k):
            pass

        def close(self):
            pass

        def display(self, *_a, **_k):
            pass

        def clear(self, *_a, **_k):
            pass

        def refresh(self, *_a, **_k):
            pass

        def reset(self, total=None):
            pass

        def write(self, s="", *_a, **_k):
            pass

        @property
        def n(self):
            return 0

        @classmethod
        def get_lock(cls):
            if not hasattr(cls, "_lock"):
                cls._lock = threading.RLock()
            return cls._lock

        @classmethod
        def set_lock(cls, lock):
            cls._lock = lock

    return _T


@contextmanager
def _byte_progress(asset: Asset, lock: threading.Lock):
    """Monkeypatch huggingface_hub's internal per-file progress-bar class so
    real byte counts reach `asset.bytes_total`/`bytes_done`.

    `_get_progress_bar_context` (huggingface_hub/utils/tqdm.py) instantiates
    the module-level `tqdm` name directly for each file's byte-level bar —
    that's the only place actual transfer sizes are known, since
    `snapshot_download`'s own `tqdm_class=` kwarg is documented as not being
    passed down to individual file downloads. Patched for the duration of one
    asset's download call, then restored.
    """
    # Neither `from huggingface_hub.utils import tqdm` nor `import
    # huggingface_hub.utils.tqdm as m` reach the actual submodule here:
    # `huggingface_hub/utils/__init__.py` does `from .tqdm import tqdm`, which
    # overwrites the `tqdm` *attribute* on the `utils` package with the class —
    # and `import a.b.c as x` walks attributes, not `sys.modules`, so it
    # resolves to that same class. Go through `sys.modules` directly to get the
    # real submodule whose global `tqdm` name `_get_progress_bar_context` reads.
    import sys as _sys

    import huggingface_hub.utils.tqdm  # noqa: F401 — ensure it's imported/cached

    _hf_tqdm_module = _sys.modules["huggingface_hub.utils.tqdm"]

    class _P:
        def __init__(self, *, total=None, initial=0, **_kw):
            with lock:
                if total:
                    asset.bytes_total = (asset.bytes_total or 0) + total
                if initial:
                    asset.bytes_done += initial

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def update(self, n=1):
            with lock:
                asset.bytes_done += n

        def close(self):
            pass

        def set_description(self, *_a, **_k):
            pass

        def set_postfix(self, **_):
            pass

        def refresh(self, *_a, **_k):
            pass

    original = _hf_tqdm_module.tqdm
    _hf_tqdm_module.tqdm = _P
    try:
        yield
    finally:
        _hf_tqdm_module.tqdm = original


def _default_snapshot(repo: str, dest: str, allow=None, revision=None) -> None:
    from huggingface_hub import snapshot_download

    Path(dest).mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, cache_dir=dest, allow_patterns=allow, revision=revision)


def _default_snapshot_with_progress(
    repo: str, dest: str, allow, revision, asset: Asset, lock: threading.Lock
) -> None:
    from huggingface_hub import snapshot_download

    Path(dest).mkdir(parents=True, exist_ok=True)
    with _byte_progress(asset, lock):
        snapshot_download(
            repo_id=repo,
            cache_dir=dest,
            allow_patterns=allow,
            revision=revision,
            tqdm_class=_make_asset_tqdm(asset, lock),
        )


def _default_file(repo: str, filename: str, dest: str) -> None:
    from huggingface_hub import hf_hub_download

    Path(dest).mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo_id=repo, filename=filename, local_dir=dest)


def _default_file_with_progress(
    repo: str, filename: str, dest: str, asset: Asset, lock: threading.Lock
) -> None:
    from huggingface_hub import hf_hub_download

    Path(dest).mkdir(parents=True, exist_ok=True)
    with _byte_progress(asset, lock):
        hf_hub_download(repo_id=repo, filename=filename, local_dir=dest)


def _default_dir_snapshot(repo: str, dest: str, allow=None) -> None:
    # Snapshot the repo *flat* into dest (not the hub cache) so a directory-style
    # model (e.g. the ONNX bundle) lands where the loader reads it. `allow`
    # (allow_patterns) fetches only the needed files from a large repo.
    from huggingface_hub import snapshot_download

    Path(dest).mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=allow)


def _default_dir_snapshot_with_progress(
    repo: str, dest: str, allow, asset: Asset, lock: threading.Lock
) -> None:
    from huggingface_hub import snapshot_download

    Path(dest).mkdir(parents=True, exist_ok=True)
    with _byte_progress(asset, lock):
        snapshot_download(
            repo_id=repo,
            local_dir=dest,
            allow_patterns=allow,
            tqdm_class=_make_asset_tqdm(asset, lock),
        )


def _default_url(url: str, dest: str) -> None:
    import urllib.request

    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _default_url_with_progress(url: str, dest: str, asset: Asset, lock: threading.Lock) -> None:
    import urllib.request

    Path(dest).parent.mkdir(parents=True, exist_ok=True)

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        with lock:
            if total_size > 0 and asset.bytes_total is None:
                asset.bytes_total = total_size
            asset.bytes_done = min(
                block_num * block_size,
                total_size if total_size > 0 else block_num * block_size,
            )

    urllib.request.urlretrieve(url, dest, reporthook=_hook)


@contextmanager
def _hf_online():
    """Force huggingface_hub online for the duration of a download.

    `HF_HUB_OFFLINE` is cached by `huggingface_hub.constants` as a plain
    module-level bool at *import* time (see `constants.py`), so once that
    module has been imported anywhere in this process — e.g. an earlier
    download, or a model load — flipping `os.environ["HF_HUB_OFFLINE"]`
    afterward has no effect on it. Every offline check in the library reads
    `constants.HF_HUB_OFFLINE` back off that same module object though (not a
    `from constants import HF_HUB_OFFLINE` copy), so patching the attribute
    directly does take effect immediately, import order notwithstanding.
    Restores both the attribute and the env var on exit so the runtime goes
    back to offline once the download finishes.
    """
    import huggingface_hub.constants as hf_constants
    from huggingface_hub import configure_http_backend

    prev_env = os.environ.get("HF_HUB_OFFLINE")
    prev_attr = hf_constants.HF_HUB_OFFLINE
    os.environ["HF_HUB_OFFLINE"] = "0"
    hf_constants.HF_HUB_OFFLINE = False
    # huggingface_hub caches one requests session per thread. A prior offline
    # model load leaves those sessions mounted with OfflineAdapter even after
    # the constant above is changed, so rebuild them for this wizard download.
    configure_http_backend()
    try:
        yield
    finally:
        hf_constants.HF_HUB_OFFLINE = prev_attr
        if prev_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_env
        configure_http_backend()


def _already_present(asset: Asset) -> bool:
    """Skip an asset already on disk (cheap resume — hf also resumes internally)."""
    if asset.kind == "url":
        return Path(asset.dest).is_file()
    if asset.kind == "file":
        return (Path(asset.dest) / asset.filename).is_file()
    if asset.kind == "dir_snapshot":
        d = Path(asset.dest)
        if not d.is_dir():
            return False
        # Sentinel is written on successful completion, but a killed/interrupted
        # run can still leave it behind with no actual model files underneath
        # (e.g. the process died between mkdir+touch and the transfer). Require
        # at least one non-sentinel file too, so a sentinel-only dir is treated
        # as missing and re-downloaded rather than silently loaded as complete.
        return any(f for f in d.iterdir() if f.name != COMPLETE_SENTINEL)
    # Snapshot downloads created by this version carry an explicit completion
    # marker. Retain the standard HF ref/snapshot check for pre-marker caches.
    hub_dir = Path(asset.dest) / f"models--{asset.repo.replace('/', '--')}"
    if (hub_dir / COMPLETE_SENTINEL).is_file():
        return True
    ref = hub_dir / "refs" / "main"
    try:
        revision = ref.read_text().strip()
    except OSError:
        return False
    snapshot = hub_dir / "snapshots" / revision
    return bool(revision) and snapshot.is_dir() and any(snapshot.iterdir())


class DownloadManager:
    """Runs a planned asset list on a background thread; exposes live progress."""

    def __init__(self, snapshot_fn=None, file_fn=None, url_fn=None, dir_snapshot_fn=None):
        self._snapshot_fn: Callable = snapshot_fn or _default_snapshot
        self._file_fn: Callable = file_fn or _default_file
        self._url_fn: Callable = url_fn or _default_url
        self._dir_snapshot_fn: Callable = dir_snapshot_fn or _default_dir_snapshot
        self._lock = threading.Lock()
        self._assets: list = []
        self._state = "idle"  # idle | downloading | done | error
        self._error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def active(self) -> bool:
        return self._state == "downloading"

    def start(self, assets: list, on_complete: Optional[Callable[[bool], None]] = None) -> bool:
        """Begin downloading `assets` on a daemon thread. No-op if already active."""
        with self._lock:
            if self._state == "downloading":
                return False
            self._assets = assets
            for a in assets:
                a.status = PENDING
                a.error = None
                a.needs_hf_token = False
                a.bytes_done = 0
                a.bytes_total = None
            self._state = "downloading"
            self._error = None
        self._thread = threading.Thread(
            target=self._run, args=(on_complete,), daemon=True, name="downloader"
        )
        self._thread.start()
        return True

    def _set(
        self, asset: Asset, status: str, error: Optional[str] = None, needs_hf_token: bool = False
    ) -> None:
        with self._lock:
            asset.status = status
            asset.error = error
            asset.needs_hf_token = needs_hf_token

    def _run(self, on_complete) -> None:
        ok = True
        with _hf_online():
            ok = self._download_all()
        with self._lock:
            self._state = DONE if ok else ERROR
            if not ok:
                self._error = next(
                    (a.error for a in self._assets if a.status == ERROR), "download failed"
                )
        if on_complete is not None:
            try:
                on_complete(ok)
            except Exception:
                logger.exception("download on_complete hook raised")

    def _download_all(self) -> bool:
        ok = True
        for asset in self._assets:
            try:
                if _already_present(asset):
                    self._set(asset, DONE)
                    continue
                self._set(asset, DOWNLOADING)
                logger.info("Downloading %s (%s)", asset.label, asset.key)
                if asset.kind == "snapshot":
                    if self._snapshot_fn is _default_snapshot:
                        _default_snapshot_with_progress(
                            asset.repo, asset.dest, asset.allow, asset.revision, asset, self._lock
                        )
                    else:
                        if asset.allow or asset.revision:
                            self._snapshot_fn(asset.repo, asset.dest, asset.allow, asset.revision)
                        else:
                            self._snapshot_fn(asset.repo, asset.dest)
                    hub_dir = Path(asset.dest) / f"models--{asset.repo.replace('/', '--')}"
                    hub_dir.mkdir(parents=True, exist_ok=True)
                    (hub_dir / COMPLETE_SENTINEL).touch()
                elif asset.kind == "dir_snapshot":
                    if self._dir_snapshot_fn is _default_dir_snapshot:
                        _default_dir_snapshot_with_progress(
                            asset.repo, asset.dest, asset.allow, asset, self._lock
                        )
                    else:
                        self._dir_snapshot_fn(asset.repo, asset.dest, asset.allow)
                    # Write completion sentinel so interrupted downloads are detected
                    dest = Path(asset.dest)
                    dest.mkdir(parents=True, exist_ok=True)
                    (dest / COMPLETE_SENTINEL).touch()
                elif asset.kind == "file":
                    if self._file_fn is _default_file:
                        _default_file_with_progress(
                            asset.repo, asset.filename, asset.dest, asset, self._lock
                        )
                    else:
                        self._file_fn(asset.repo, asset.filename, asset.dest)
                elif asset.kind == "url":
                    if self._url_fn is _default_url:
                        _default_url_with_progress(asset.url, asset.dest, asset, self._lock)
                    else:
                        self._url_fn(asset.url, asset.dest)
                else:
                    raise ValueError(f"unknown asset kind {asset.kind!r}")
                self._set(asset, DONE)
            except Exception as e:  # noqa: BLE001 — surface, don't crash the thread
                logger.exception("Download failed: %s", asset.key)
                self._set(
                    asset,
                    ERROR,
                    f"{type(e).__name__}: {e}",
                    needs_hf_token=_hf_access_denied(e),
                )
                ok = False
                break
        return ok

    def snapshot(self) -> dict:
        """Current overall + per-asset state, for `/status` and the SSE stream."""
        with self._lock:
            assets = [a.snapshot() for a in self._assets]
            done = sum(1 for a in assets if a["status"] == DONE)
            return {
                "state": self._state,
                "error": self._error,
                "needs_hf_token": any(a.needs_hf_token for a in self._assets),
                "completed": done,
                "total": len(assets),
                "assets": assets,
            }
