"""One-time fetch for the CrispASR-hosted Qwen3-TTS GGUF backends
(core/tts_crispasr.py, backend ids `crispasr-qwen3-tts` [1.7B] and
`crispasr-qwen3-tts-0.6b` [0.6B]).

Not wired into the setup wizard yet — these backends are experimental/
selectable, so fetching is manual for now. Pulls three things:

  1. The CrispASR Python runtime (ctypes binding + bundled libcrispasr.so and
     its ggml/whisper .so dependencies) from a pinned GitHub release —
     extracted to data/models/crispasr-python/. Shared by both talker sizes.
     Pass --gpu to also (or instead) assemble a CUDA-enabled copy at
     data/models/crispasr-python-cuda/ — see fetch_lib_gpu()'s docstring for
     why that needs assembling rather than a straight download.
   2. The requested talker GGUFs (1.7B F16, 0.6B q8_0):
       1.7B: huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF
       0.6B: huggingface.co/cstr/qwen3-tts-0.6b-base-GGUF
  3. The companion codec GGUF (fixed at f16 regardless of talker size —
     quantizing it hurts more than the talker does) from
     huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF. Downloaded once,
     symlinked into whichever talker dir(s) are fetched.

1.7B lands in data/models/qwen3-tts-crispasr-gguf/, 0.6B in
data/models/qwen3-tts-crispasr-0.6b-gguf/ (core/tts_crispasr.py's
DEFAULT_MODEL_DIR / DEFAULT_MODEL_DIR_0_6B).

Usage:
    .venv/bin/python scripts/fetch_crispasr_tts.py                # both talkers, CPU runtime (default)
    .venv/bin/python scripts/fetch_crispasr_tts.py --gpu          # + CUDA-enabled runtime (needs an NVIDIA GPU + driver)
    .venv/bin/python scripts/fetch_crispasr_tts.py --model 1.7b   # just the 1.7B
    .venv/bin/python scripts/fetch_crispasr_tts.py --model 0.6b   # just the 0.6B
    .venv/bin/python scripts/fetch_crispasr_tts.py --force        # re-fetch even if present
"""

import argparse
import logging
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fetch_crispasr_tts")

MODELS_ROOT = Path("data/models")
LIB_DIR = MODELS_ROOT / "crispasr-python"
LIB_DIR_GPU = MODELS_ROOT / "crispasr-python-cuda"

CRISPASR_RELEASE_TAG = "v0.8.7"  # pinned to the version validated during backend selection
_ARCH_TARBALLS = {
    "x86_64": "crispasr-python-linux-x86_64.tar.gz",
    "amd64": "crispasr-python-linux-x86_64.tar.gz",
    "aarch64": "crispasr-python-linux-arm64.tar.gz",
    "arm64": "crispasr-python-linux-arm64.tar.gz",
}
# CUDA build is x86_64-only (no arm64 CUDA release exists for CrispASR).
_CUDA_TARBALL = "libcrispasr-linux-x86_64-cuda.tar.gz"

CODEC_REPO = "cstr/qwen3-tts-tokenizer-12hz-GGUF"
CODEC_FILE = "qwen3-tts-tokenizer-12hz.gguf"

# model key -> (talker repo, talker filename, dest dir)
TALKERS = {
    "1.7b": (
        "cstr/qwen3-tts-1.7b-base-GGUF",
        "qwen3-tts-12hz-1.7b-base-f16.gguf",
        MODELS_ROOT / "qwen3-tts-crispasr-gguf",
    ),
    "0.6b": (
        "cstr/qwen3-tts-0.6b-base-GGUF",
        "qwen3-tts-12hz-0.6b-base-q8_0.gguf",
        MODELS_ROOT / "qwen3-tts-crispasr-0.6b-gguf",
    ),
}


def fetch_lib(force: bool = False):
    if (LIB_DIR / "crispasr" / "__init__.py").is_file() and not force:
        logger.info("SKIP crispasr Python runtime (exists): %s", LIB_DIR)
        return
    machine = platform.machine().lower()
    tarball = _ARCH_TARBALLS.get(machine)
    if tarball is None:
        raise RuntimeError(
            f"No known CrispASR Python release for platform.machine()={machine!r}; "
            f"known: {sorted(_ARCH_TARBALLS)}"
        )
    url = (
        f"https://github.com/CrispStrobe/CrispASR/releases/download/"
        f"{CRISPASR_RELEASE_TAG}/{tarball}"
    )
    logger.info("Downloading %s …", url)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archive = tmp / tarball
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tf:
            tf.extractall(tmp)
        # Release tarball is a single top-level dir (crispasr-python-linux-*/crispasr/…);
        # flatten so LIB_DIR/crispasr/__init__.py is the final layout.
        extracted = next(p for p in tmp.iterdir() if p.is_dir() and p.name.startswith("crispasr-python"))
        LIB_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(LIB_DIR, ignore_errors=True)
        shutil.move(str(extracted), str(LIB_DIR))
    logger.info("CrispASR Python runtime -> %s", LIB_DIR)


def fetch_lib_gpu(force: bool = False):
    """Assemble a CUDA-enabled crispasr Python package.

    CrispASR doesn't publish a CUDA build of the *Python* binding — only the
    CLI/C-library release (`libcrispasr-linux-x86_64-cuda.tar.gz`), which has
    the native .so's (including libggml-cuda) but not the ctypes glue
    (_binding.py/__init__.py) or libcrispasr_helpers.so (python-binding-only,
    no CUDA-specific build exists). Verified by hand (2026-07-05, RTX 5060 Ti):
    those two pieces are backend-agnostic, so this hybrid — CPU release's
    Python glue + CUDA release's native libs — loads cleanly and genuinely
    dispatches to the GPU (~3.3GB VRAM observed synthesising the 1.7B q8_0
    talker, RTF ~0.6x vs the CPU build's ~1.5-2.5x).

    The two releases use different SONAMEs for the shared ggml libs (CPU:
    bare `libggml.so`; CUDA: versioned `libggml.so.0`) — libcrispasr.so's own
    NEEDED entries expect the CUDA build's versioned names, so the extracted
    files are renamed to match rather than the bare names fetch_lib() uses.
    """
    fetch_lib(force=force)  # need its Python glue + libcrispasr_helpers.so
    dst = LIB_DIR_GPU / "crispasr"
    if (dst / "libggml-cuda.so.0").is_file() and not force:
        logger.info("SKIP crispasr CUDA runtime (exists): %s", LIB_DIR_GPU)
        return

    url = (
        f"https://github.com/CrispStrobe/CrispASR/releases/download/"
        f"{CRISPASR_RELEASE_TAG}/{_CUDA_TARBALL}"
    )
    logger.info("Downloading %s …", url)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archive = tmp / _CUDA_TARBALL
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tf:
            tf.extractall(tmp)
        extracted = next(p for p in tmp.iterdir() if p.is_dir() and p.name.startswith("libcrispasr-linux"))

        dst.mkdir(parents=True, exist_ok=True)
        for name in ("_binding.py", "__init__.py", "libcrispasr_helpers.so"):
            shutil.copy2(LIB_DIR / "crispasr" / name, dst / name)
        # (source relative path under the CUDA tarball, dest filename with the SONAME it needs)
        for src_rel, dst_name in (
            ("src/libcrispasr.so.0.8.7", "libcrispasr.so"),
            ("ggml/src/libggml.so.0.10.2", "libggml.so.0"),
            ("ggml/src/libggml-cpu.so.0.10.2", "libggml-cpu.so.0"),
            ("ggml/src/libggml-base.so.0.10.2", "libggml-base.so.0"),
            ("ggml/src/libggml-cuda.so.0.10.2", "libggml-cuda.so.0"),
        ):
            shutil.copy2(extracted / src_rel, dst / dst_name)
    logger.info("CrispASR CUDA runtime -> %s", LIB_DIR_GPU)


def fetch_gguf(models: list, force: bool = False):
    from huggingface_hub import hf_hub_download

    # Download the shared codec once, into the first requested talker's dir,
    # then symlink it into any other requested talker dir(s).
    dest_dirs = [TALKERS[m][2] for m in models]
    dest_dirs[0].mkdir(parents=True, exist_ok=True)
    codec_master = dest_dirs[0] / CODEC_FILE
    if not codec_master.is_file() or force:
        logger.info("Downloading %s from %s …", CODEC_FILE, CODEC_REPO)
        hf_hub_download(repo_id=CODEC_REPO, filename=CODEC_FILE, local_dir=str(dest_dirs[0]))
    else:
        logger.info("SKIP %s (exists)", CODEC_FILE)
    for dest in dest_dirs[1:]:
        dest.mkdir(parents=True, exist_ok=True)
        link = dest / CODEC_FILE
        if not link.exists() and not link.is_symlink():
            link.symlink_to(Path("..") / dest_dirs[0].name / CODEC_FILE)

    for model in models:
        repo, filename, dest = TALKERS[model]
        dst = dest / filename
        if dst.is_file() and not force:
            logger.info("SKIP %s (exists)", filename)
            continue
        logger.info("Downloading %s from %s …", filename, repo)
        hf_hub_download(repo_id=repo, filename=filename, local_dir=str(dest))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("1.7b", "0.6b", "both"), default="both")
    ap.add_argument("--gpu", action="store_true", help="also assemble the CUDA-enabled runtime")
    ap.add_argument("--force", action="store_true", help="re-fetch even if already present")
    args = ap.parse_args()
    models = list(TALKERS) if args.model == "both" else [args.model]

    fetch_lib(force=args.force)
    if args.gpu:
        fetch_lib_gpu(force=args.force)
    fetch_gguf(models, force=args.force)
    backend_ids = {"1.7b": "qwen-gguf", "0.6b": "qwen-gguf-small"}
    logger.info(
        "Done. Select backend %s (models.tts.backend) to use it%s.",
        " or ".join(backend_ids[m] for m in models),
        " with opts: {gpu: true} to use the CUDA runtime" if args.gpu else "",
    )


if __name__ == "__main__":
    main()
