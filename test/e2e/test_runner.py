"""Runner tests 0.1, 0.3, 0.4, 0.8 plus exit-code mapping, with stub stages (no docker, no GPU).

Run: python3 test/e2e/test_runner.py
Everything happens in a temp directory; the disk floor is lowered because the Mac may be nearly full.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "pipeline" / "remote" / "ns_remote.py"
STUBS = ROOT / "test" / "e2e" / "stubs"
sys.path.insert(0, str(ROOT))

STAGE_JSON = {"ingest": "10_frames/frames.json", "sfm": "20_sfm/sfm.json", "align": "40_align/align.json",
              "train": "50_train/train.json", "geometry": "55_geometry/geometry.json",
              "export": "60_export/export.json", "grid": "70_grid/grid.json"}


def runner_cmd(root: Path, run: str, *extra: str, stubs: Path = STUBS) -> list:
    return [sys.executable, str(RUNNER), run, "--runs-root", str(root), "--no-docker", "--stub-dir", str(stubs),
            "--min-free-gb", "0.05", *extra]


def make_run(root: Path, name: str, cfg: dict | None = None, n_frames: int = 3) -> Path:
    run = root / name
    (run / "input" / "frames").mkdir(parents=True)
    for i in range(n_frames):
        (run / "input" / "frames" / f"{i:03d}.jpg").write_bytes(b"x")
    (run / "input" / "meta.json").write_text(json.dumps({"height_cm": 175}))
    if cfg:
        (run / "stub_cfg.json").write_text(json.dumps(cfg))
    return run


def calls(run: Path) -> list:
    p = run / "stub_calls.log"
    return [l.split() for l in p.read_text().splitlines()] if p.exists() else []


def starts(run: Path, stage: str) -> int:
    return sum(1 for c in calls(run) if c[0] == stage and c[1] == "start")


def wait_for(pred, timeout=20.0, step=0.1):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(step)
    return False


def status(run: Path) -> dict:
    try:
        return json.loads((run / "status.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


class RunnerBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rhrunner"))
        self.procs: list = []

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_runner(self, name, *extra, timeout=60, **kw):
        return subprocess.run(runner_cmd(self.tmp, name, *extra, **kw), capture_output=True, text=True, timeout=timeout)


class FullRun(RunnerBase):
    def test_all_stages_run_in_order_and_write_files(self):
        run = make_run(self.tmp, "full")
        r = self.run_runner("full", "--quick", "--git", "abc123")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for stage, rel in STAGE_JSON.items():
            self.assertTrue((run / rel).exists(), rel)
        order = [c[0] for c in calls(run) if c[1] == "start"]
        self.assertEqual(order, ["ingest", "sfm", "align", "train", "geometry", "export", "grid"])
        t = json.loads((run / "timings.json").read_text())
        self.assertEqual(set(t["stages"]), set(STAGE_JSON))
        rj = json.loads((run / "run.json").read_text())
        self.assertEqual((rj["git"], rj["exit"], rj["flags"]["quick"]), ("abc123", 0, True))
        self.assertEqual(json.loads((run / "50_train" / "train.json").read_text())["variants"][0]["steps"], 7000)
        self.assertEqual(status(run)["state"], "done")
        self.assertTrue((run / "logs" / "train.log").exists())

    def test_second_run_skips_everything(self):
        run = make_run(self.tmp, "again")
        self.assertEqual(self.run_runner("again").returncode, 0)
        before = {rel: (run / rel).stat().st_mtime_ns for rel in STAGE_JSON.values()}
        r = self.run_runner("again")
        self.assertEqual(r.returncode, 0)
        self.assertEqual(before, {rel: (run / rel).stat().st_mtime_ns for rel in STAGE_JSON.values()})
        self.assertEqual(starts(run, "ingest"), 1)
        self.assertIn("done already", r.stdout)

    def test_to_limits_the_stage_list(self):
        run = make_run(self.tmp, "to")
        self.assertEqual(self.run_runner("to", "--to", "align").returncode, 0)
        self.assertTrue((run / "40_align" / "align.json").exists())
        self.assertFalse((run / "50_train").exists())

    def test_run_name_is_validated(self):
        r = self.run_runner("Bad Name")
        self.assertNotEqual(r.returncode, 0)


class Test0_1_KillAndResume(RunnerBase):
    def _start_and_wait_for_train(self, name):
        run = make_run(self.tmp, name, {"train": {"sleep": 60}})
        p = subprocess.Popen(runner_cmd(self.tmp, name), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs.append(p)
        self.assertTrue(wait_for(lambda: status(run).get("stages", {}).get("train", {}).get("state") == "running"),
                        "train never started")
        return run, p

    def test_sigterm_during_training_then_from_train(self):
        run, p = self._start_and_wait_for_train("k1")
        mt = {rel: (run / rel).stat().st_mtime_ns for rel in ("10_frames/frames.json", "20_sfm/sfm.json", "40_align/align.json")}
        p.send_signal(signal.SIGTERM)
        out, _ = p.communicate(timeout=20)
        self.assertEqual(p.returncode, 20, out)
        self.assertFalse((run / "50_train" / "train.json").exists())
        self.assertEqual(status(run)["state"], "stopped")
        # restart with --from train and a fast train stage
        (run / "stub_cfg.json").write_text("{}")
        r = self.run_runner("k1", "--from", "train")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((run / "50_train" / "train.json").exists())
        for rel, t in mt.items():
            self.assertEqual((run / rel).stat().st_mtime_ns, t, f"{rel} was rewritten")
        for stage in ("ingest", "sfm", "align"):
            self.assertEqual(starts(run, stage), 1, stage)
        self.assertEqual(starts(run, "train"), 2)

    def test_sigkill_runner_and_child_then_plain_resume(self):
        run, p = self._start_and_wait_for_train("k2")
        # kill -9 the runner and then its process group child, as a crash would
        child_cmds = subprocess.run(["pgrep", "-f", str(STUBS / "train.py")], capture_output=True, text=True).stdout.split()
        p.kill()
        p.wait()
        for pid in child_cmds:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
        (run / "stub_cfg.json").write_text("{}")
        r = self.run_runner("k2")  # no --from: skips done stages by their JSON
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((run / "50_train" / "train.json").exists())
        self.assertFalse((run / "50_train.tmp").exists())
        for stage in ("ingest", "sfm", "align"):
            self.assertEqual(starts(run, stage), 1, stage)

    def test_from_reruns_that_stage_and_later_ones(self):
        run = make_run(self.tmp, "from")
        self.assertEqual(self.run_runner("from").returncode, 0)
        self.assertEqual(self.run_runner("from", "--from", "export").returncode, 0)
        self.assertEqual([starts(run, s) for s in ("ingest", "sfm", "align", "train", "export", "grid")], [1, 1, 1, 1, 2, 2])


class Test0_3_LowDisk(RunnerBase):
    def test_fake_low_disk_exits_20_within_5_seconds(self):
        run = make_run(self.tmp, "disk")
        t0 = time.monotonic()
        r = self.run_runner("disk", "--fake-free-gb", "3", "--min-free-gb", "15")
        dt = time.monotonic() - t0
        self.assertEqual(r.returncode, 20, r.stdout + r.stderr)
        self.assertLess(dt, 5.0)
        self.assertFalse((run / "10_frames" / "frames.json").exists())
        self.assertIn("disk guard", r.stdout + r.stderr)

    def test_stage_level_guard_also_exits_20(self):
        """A stage using stage_context sees NS_FAKE_FREE_GB; the runner floor is lowered so the stage decides."""
        run = make_run(self.tmp, "disk2")
        # the stubs lower their own floor, so go through the real helper's CLI
        r = subprocess.run([sys.executable, str(ROOT / "pipeline" / "common" / "stage.py"), "guard", str(run)],
                           env=dict(os.environ, NS_FAKE_FREE_GB="2"), capture_output=True, text=True, timeout=10)
        self.assertEqual(r.returncode, 20)


class Test0_4_NoCollision(RunnerBase):
    def test_two_runs_at_once(self):
        a = make_run(self.tmp, "rh-a", {"train": {"sleep": 1.5}})
        b = make_run(self.tmp, "rh-b", {"train": {"sleep": 1.5}}, n_frames=5)
        pa = subprocess.Popen(runner_cmd(self.tmp, "rh-a"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        pb = subprocess.Popen(runner_cmd(self.tmp, "rh-b"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs += [pa, pb]
        oa, _ = pa.communicate(timeout=60)
        ob, _ = pb.communicate(timeout=60)
        self.assertEqual((pa.returncode, pb.returncode), (0, 0), oa + ob)
        na = json.loads((a / "10_frames" / "frames.json").read_text())["n_kept"]
        nb = json.loads((b / "10_frames" / "frames.json").read_text())["n_kept"]
        self.assertEqual((na, nb), (3, 5))
        # the runs overlapped in time
        ta = [float(c[2]) for c in calls(a) if c[0] == "train"]
        tb = [float(c[2]) for c in calls(b) if c[0] == "train"]
        self.assertLess(max(ta[0], tb[0]), min(ta[1], tb[1]))

    def test_same_name_twice_is_refused(self):
        run = make_run(self.tmp, "dup", {"train": {"sleep": 30}})
        p = subprocess.Popen(runner_cmd(self.tmp, "dup"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs.append(p)
        self.assertTrue(wait_for(lambda: status(run).get("stages", {}).get("train", {}).get("state") == "running"))
        r = self.run_runner("dup")
        self.assertEqual(r.returncode, 20)
        self.assertIn("already", r.stdout)
        p.send_signal(signal.SIGTERM)
        p.communicate(timeout=20)


class Test0_8_Rescale(RunnerBase):
    def test_rescale_changes_grid_not_training(self):
        run = make_run(self.tmp, "resc")
        self.assertEqual(self.run_runner("resc").returncode, 0)
        train_mtime = (run / "50_train" / "train.json").stat().st_mtime_ns
        grid0 = json.loads((run / "70_grid" / "grid.json").read_text())
        t0 = time.monotonic()
        r = self.run_runner("resc", "--rescale", "--opt", "rescale:--factor=2")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertLess(time.monotonic() - t0, 60)
        grid1 = json.loads((run / "70_grid" / "grid.json").read_text())
        exp = json.loads((run / "60_export" / "export.json").read_text())
        self.assertEqual(exp["T_final"][0][0], 2.0)
        self.assertAlmostEqual(grid1["room_extent_m"][0], 2 * grid0["room_extent_m"][0])
        self.assertEqual(starts(run, "train"), 1)
        self.assertEqual(starts(run, "export"), 1)
        self.assertEqual(starts(run, "grid"), 2)
        self.assertEqual((run / "50_train" / "train.json").stat().st_mtime_ns, train_mtime)

    def test_rescale_without_export_does_nothing(self):
        make_run(self.tmp, "resc2")
        r = self.run_runner("resc2", "--rescale")
        self.assertEqual(r.returncode, 20)


class ExitCodes(RunnerBase):
    def test_capture_problem_stops_with_10(self):
        run = make_run(self.tmp, "e10", {"ingest": {"exit": 10}})
        r = self.run_runner("e10")
        self.assertEqual(r.returncode, 10, r.stdout)
        self.assertFalse((run / "20_sfm").exists())
        self.assertEqual(json.loads((run / "run.json").read_text())["exit"], 10)

    def test_partial_and_gate_continue_and_combine(self):
        run = make_run(self.tmp, "e11", {"sfm": {"exit": 11}})
        # the stub writes no JSON on non-zero exit, so a 11 without a marker counts as a crash (20)
        r = self.run_runner("e11")
        self.assertEqual(r.returncode, 20)
        self.assertIn("was not written", json.loads((run / "run.json").read_text())["warnings"][0])

    def test_ns_result_gate_failed_gives_30_and_the_run_continues(self):
        run = make_run(self.tmp, "g30", {"align": {"gate_failed": True}})
        r = self.run_runner("g30")
        self.assertEqual(r.returncode, 30, r.stdout)
        self.assertTrue((run / "70_grid" / "grid.json").exists())  # later stages still ran
        rj = json.loads((run / "run.json").read_text())
        self.assertEqual(rj["exit"], 30)
        self.assertTrue(any("quality gate failed: stub gate flag" in w for w in rj["warnings"]))
        self.assertEqual(status(run)["stages"]["align"]["state"], "exit-30")
        self.assertEqual(status(run)["state"], "review")

    def test_unknown_exit_code_maps_to_20(self):
        make_run(self.tmp, "e7", {"align": {"exit": 7}})
        r = self.run_runner("e7")
        self.assertEqual(r.returncode, 20)

    def test_failed_forced_stage_leaves_the_stage_not_done(self):
        """Stale results must not come back: after a failed --from run, the stage has no JSON and the next plain run redoes it."""
        run = make_run(self.tmp, "keep")
        self.assertEqual(self.run_runner("keep").returncode, 0)
        (run / "stub_cfg.json").write_text(json.dumps({"train": {"exit": 20}}))
        r = self.run_runner("keep", "--from", "train")
        self.assertEqual(r.returncode, 20)
        self.assertFalse((run / "50_train" / "train.json").exists())
        self.assertFalse((run / "50_train.prev").exists())
        (run / "stub_cfg.json").write_text("{}")
        self.assertEqual(self.run_runner("keep").returncode, 0)
        self.assertTrue((run / "50_train" / "train.json").exists())
        self.assertEqual(starts(run, "ingest"), 1)

    def test_failure_with_a_json_written_does_not_look_done(self):
        run = make_run(self.tmp, "fj")
        # a stage that writes its JSON and then exits 20
        stubs = self.tmp / "stubs"
        shutil.copytree(STUBS, stubs)
        (stubs / "sfm.py").write_text(
            "import sys, json\nfrom pathlib import Path\nrun = Path(sys.argv[sys.argv.index('--run') + 1])\n"
            "(run / '20_sfm').mkdir(exist_ok=True)\n(run / '20_sfm' / 'sfm.json').write_text('{}')\nsys.exit(20)\n")
        r = self.run_runner("fj", stubs=stubs)
        self.assertEqual(r.returncode, 20)
        self.assertFalse((run / "20_sfm" / "sfm.json").exists())
        self.assertTrue((run / "20_sfm" / "sfm.json.failed").exists())


class MissingModules(RunnerBase):
    def _stubs_without(self, *names) -> Path:
        d = self.tmp / "stubs"
        shutil.copytree(STUBS, d)
        for n in names:
            (d / f"{n}.py").unlink()
        return d

    def test_missing_module_is_skipped_with_message_not_crash(self):
        run = make_run(self.tmp, "miss")
        stubs = self._stubs_without("train", "export", "geometry")
        r = self.run_runner("miss", stubs=stubs)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 20, out)   # incomplete: required modules are missing
        self.assertIn("SKIP train: module pipeline.train not found", out)
        self.assertIn("INCOMPLETE", out)
        self.assertNotIn("Traceback", out)
        self.assertTrue((run / "40_align" / "align.json").exists())
        st = status(run)["stages"]
        self.assertEqual(st["train"]["state"], "skipped-missing")
        self.assertEqual(st["export"]["state"], "skipped-missing")
        self.assertEqual(st["grid"]["state"], "skipped-needs")

    def test_optional_module_missing_is_fine_and_allow_missing_exits_0(self):
        make_run(self.tmp, "miss2")
        stubs = self._stubs_without("geometry")  # optional stage only
        r = self.run_runner("miss2", stubs=stubs)
        self.assertEqual(r.returncode, 0, r.stdout)
        stubs2 = self.tmp / "stubs2"
        shutil.copytree(STUBS, stubs2)
        (stubs2 / "geometry.py").unlink()
        (stubs2 / "export.py").unlink()
        (stubs2 / "grid.py").unlink()
        make_run(self.tmp, "miss4")
        self.assertEqual(self.run_runner("miss4", stubs=stubs2).returncode, 20)
        self.assertEqual(self.run_runner("miss4", "--allow-missing", stubs=stubs2).returncode, 0)

    def test_force_does_not_destroy_outputs_of_missing_module(self):
        run = make_run(self.tmp, "miss3")
        self.assertEqual(self.run_runner("miss3").returncode, 0)
        stubs = self.tmp / "stubs2"
        shutil.copytree(STUBS, stubs)
        (stubs / "train.py").unlink()
        r = self.run_runner("miss3", "--from", "train", stubs=stubs)
        self.assertTrue((run / "50_train" / "train.json").exists())
        self.assertEqual(r.returncode, 20)


if __name__ == "__main__":
    unittest.main()
