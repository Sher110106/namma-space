"""Helpers every stage can use, plus the stage table the runner and the report read.

Standard library only. Copy-import friendly.

A stage is `python -m pipeline.<stage> --run <run_dir> [opts]`. It reads earlier stages'
JSON, writes only its own directory, and is done when its JSON exists (CONTRACTS 2.11).

Typical use inside a stage module
    from pipeline.common import stage as st

    def main(argv=None):
        run = st.Run(args.run)
        with st.stage_context(run, "train") as ctx:      # disk guard + timing + log + atomic dir
            align = run.read_json("40_align/align.json")
            ... write files under ctx.out ...
            run.write_json(ctx.out / "train.json", {...})  # or st.write_json_atomic
        # leaving the block renames 50_train.tmp to 50_train
    if __name__ == "__main__":
        st.run_main(main)            # maps StageExit(code) and crashes to exit codes

API
    EXIT_OK, EXIT_CAPTURE, EXIT_PARTIAL, EXIT_INFRA, EXIT_GATE      exit codes (CONTRACTS 2.11)
    StageExit(code, msg)                                            raise to stop with a code
    atomic_stage(run_dir, dirname)                                  write to <dirname>.tmp, rename on success
    write_json_atomic(path, obj), read_json(path, default)
    Run(run_dir)                                                    .path(rel) .read_json(rel) .done(stage)
    disk_guard(path, min_gb=15)                                     exits 20 below the floor; NS_FAKE_FREE_GB fakes it
    timed(run_dir, stage)                                           adds seconds to <run>/timings.json
    stage_context(run, stage, dirname=None, min_gb=15)              the three above plus logging
    STAGES                                                          the stage table (name, module, outputs, json, image, gpu, requires)
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

EXIT_OK = 0
EXIT_CAPTURE = 10    # capture problem; the time ranges to reshoot are printed
EXIT_PARTIAL = 11    # partial reconstruction shipped
EXIT_INFRA = 20      # infrastructure problem (disk, container, crash)
EXIT_GATE = 30       # a quality gate failed; a bundle exists but needs review
EXIT_NAMES = {0: "ok", 10: "capture", 11: "partial", 20: "infra", 30: "gate"}

MIN_FREE_GB = 15.0
FAKE_FREE_ENV = "NS_FAKE_FREE_GB"
_MISSING = object()


# ------------------------------------------------------------------ stage table

@dataclass(frozen=True)
class StageInfo:
    name: str                  # short name used by --from / --to
    module: str                # python -m <module>
    outputs: tuple             # directories the stage writes, relative to the run dir
    json: str                  # done marker, relative to the run dir
    image: str                 # docker image on tyrone
    gpu: bool
    requires: tuple = ()       # files that must exist before the stage can run
    optional: bool = False     # P1 stages: a missing module is not an error


GPU_IMAGE = "namma-space-gpu:0.1"
EXPORT_IMAGE = "namma-space-gpu:0.2"
COLMAP_IMAGE = "namma-colmap:0.2"  # 0.1 has no python3; 0.2 adds it (W1)

STAGES: tuple = (
    StageInfo("ingest", "pipeline.ingest", ("10_frames",), "10_frames/frames.json", GPU_IMAGE, False,
              requires=("input",)),
    StageInfo("sfm", "pipeline.sfm", ("20_sfm", "30_undist"), "20_sfm/sfm.json", COLMAP_IMAGE, True,
              requires=("10_frames/frames.json",)),
    StageInfo("align", "pipeline.align", ("40_align",), "40_align/align.json", GPU_IMAGE, False,
              requires=("20_sfm/sfm.json",)),
    StageInfo("train", "pipeline.train", ("50_train",), "50_train/train.json", GPU_IMAGE, True,
              requires=("40_align/align.json",)),
    StageInfo("geometry", "pipeline.geometry", ("55_geometry",), "55_geometry/geometry.json", GPU_IMAGE, True,
              requires=("50_train/train.json",), optional=True),
    StageInfo("export", "pipeline.export", ("60_export",), "60_export/export.json", EXPORT_IMAGE, True,
              requires=("50_train/train.json", "40_align/align.json")),
    StageInfo("grid", "pipeline.grid", ("70_grid",), "70_grid/grid.json", GPU_IMAGE, False,
              requires=("60_export/export.json", "40_align/align.json")),
)
STAGE_BY_NAME = {s.name: s for s in STAGES}
STAGE_NAMES = tuple(s.name for s in STAGES)


# ------------------------------------------------------------------ json and time

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json_atomic(path, obj: Any) -> None:
    """Write JSON to <path>.tmp, then rename. Readers never see half a file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n")
    os.replace(tmp, p)


def read_json(path, default: Any = _MISSING) -> Any:
    p = Path(path)
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        if default is _MISSING:
            raise
        return default
    except json.JSONDecodeError as exc:
        if default is _MISSING:
            raise ValueError(f"{p}: invalid JSON ({exc})") from exc
        return default


class StageExit(Exception):
    """Raise inside a stage to stop with one of the contract exit codes."""

    def __init__(self, code: int, msg: str = ""):
        super().__init__(msg)
        self.code = code
        self.msg = msg


# ------------------------------------------------------------------ disk guard

def free_gb(path) -> float:
    fake = os.environ.get(FAKE_FREE_ENV)
    if fake not in (None, ""):
        return float(fake)
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def disk_guard(path, min_gb: float = MIN_FREE_GB) -> float:
    """Exit 20 when less than min_gb is free on the volume holding path.

    NS_FAKE_FREE_GB=3 fakes a nearly full disk (test 0.3). Returns the free GB.
    """
    free = free_gb(path)
    if free < min_gb:
        print(f"disk guard: only {free:.1f} GB free at {path} (floor {min_gb:g} GB); stopping with exit 20",
              file=sys.stderr, flush=True)
        raise SystemExit(EXIT_INFRA)
    return free


# ------------------------------------------------------------------ run directory

class Run:
    """A run directory: runs/<name>/ (CONTRACTS 2.11)."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir)

    def path(self, rel: str) -> Path:
        return self.dir / rel

    def read_json(self, rel: str, default: Any = _MISSING) -> Any:
        return read_json(self.dir / rel, default)

    def write_json(self, rel_or_path, obj: Any) -> None:
        p = Path(rel_or_path)
        write_json_atomic(p if p.is_absolute() else self.dir / p, obj)

    def done(self, stage: str) -> bool:
        info = STAGE_BY_NAME[stage]
        return (self.dir / info.json).exists()

    def meta(self) -> dict:
        return self.read_json("input/meta.json", {})


# ------------------------------------------------------------------ atomic stage directories

@contextlib.contextmanager
def atomic_stage(run_dir, dirname: str) -> Iterator[Path]:
    """Yield <run>/<dirname>.tmp; on success rename it to <run>/<dirname>.

    A leftover .tmp from a killed run is removed first. An existing final directory is
    replaced only after the new one is complete (moved aside, then deleted). On an
    exception the .tmp is kept, named <dirname>.failed, for inspection.
    """
    base = Path(run_dir)
    final = base / dirname
    tmp = base / (dirname + ".tmp")
    failed = base / (dirname + ".failed")
    for stale in (tmp, failed):
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        yield tmp
    except BaseException:
        if tmp.exists():
            os.replace(tmp, failed)
        raise
    old = None
    if final.exists() or final.is_symlink():
        old = base / (dirname + ".old")
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        os.replace(final, old)
    os.replace(tmp, final)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


# ------------------------------------------------------------------ timing

def add_timing(run_dir, stage: str, seconds: float) -> None:
    """Record seconds for a stage in <run>/timings.json: {"stages": {name: s}, "total": s}."""
    p = Path(run_dir) / "timings.json"
    t = read_json(p, {"stages": {}, "total": 0.0})
    t.setdefault("stages", {})[stage] = round(seconds, 2)
    t["total"] = round(sum(t["stages"].values()), 2)
    t["updated"] = now_iso()
    write_json_atomic(p, t)


@contextlib.contextmanager
def timed(run_dir, stage: str) -> Iterator[None]:
    t0 = time.monotonic()
    try:
        yield
    finally:
        add_timing(run_dir, stage, time.monotonic() - t0)


# ------------------------------------------------------------------ the one-stop context

@dataclass
class StageContext:
    run: Run
    stage: str
    out: Path
    log_path: Path
    notes: dict = field(default_factory=dict)

    def log(self, msg: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} [{self.stage}] {msg}"
        print(line, flush=True)
        if not os.environ.get("NS_RUNNER"):  # under the runner, stdout already goes to this log
            with self.log_path.open("a") as fh:
                fh.write(line + "\n")


@contextlib.contextmanager
def stage_context(run, stage: str, dirname: str | None = None, min_gb: float = MIN_FREE_GB) -> Iterator[StageContext]:
    """Disk guard, timing, a log file in <run>/logs/, and an atomic output directory."""
    run = run if isinstance(run, Run) else Run(run)
    disk_guard(run.dir, min_gb)
    info = STAGE_BY_NAME.get(stage)
    dirname = dirname or (info.outputs[0] if info else stage)
    (run.dir / "logs").mkdir(parents=True, exist_ok=True)
    with timed(run.dir, stage), atomic_stage(run.dir, dirname) as out:
        ctx = StageContext(run, stage, out, run.dir / "logs" / f"{stage}.log")
        ctx.log(f"start, writing {dirname}")
        yield ctx
        ctx.log("done")


def run_main(main: Callable[[], int | None]) -> None:
    """Call main() and exit with a contract code. StageExit(code) and SystemExit pass through;
    any other exception prints a traceback and exits 20."""
    try:
        rc = main()
    except StageExit as exc:
        if exc.msg:
            print(exc.msg, file=sys.stderr, flush=True)
        sys.exit(exc.code)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(EXIT_INFRA)
    sys.exit(rc or 0)


# ------------------------------------------------------------------ CLI (used by the runner and tests)

def _cli(argv: list) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="stage helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("guard", help="exit 20 if less than --min-gb is free at PATH")
    g.add_argument("path")
    g.add_argument("--min-gb", type=float, default=MIN_FREE_GB)
    s = sub.add_parser("stages", help="print the stage table as JSON")
    args = ap.parse_args(argv)
    if args.cmd == "guard":
        print(f"{disk_guard(args.path, args.min_gb):.1f} GB free")
        return 0
    print(json.dumps([s.__dict__ for s in STAGES], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
