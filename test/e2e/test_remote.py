"""Tests 0.1 (via ns), 0.2, 0.4 and 0.5 against tyrone. Skipped unless ssh works and NS_E2E_REMOTE=1.

    NS_E2E_REMOTE=1 python3 test/e2e/test_remote.py
Stages are stubs run with the host python (--no-docker), so the tests do not wait for container or GPU slots.
Set NS_E2E_DOCKER=1 to add one run through nsdocker with the stubs in real containers (may queue behind other builders).
Everything lives under runs/rh-e2e-* on tyrone and is deleted at the end.
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
NS = str(ROOT / "ns")
HOST = os.environ.get("NS_HOST", "sher@10.1.18.190")
RROOT = os.environ.get("NS_REMOTE_ROOT", "/home/sher/namma-space")
FRAMES = ROOT / "test" / "data" / "honka_subset" / "frames"
STUB_FLAG = "--no-docker --stub-dir test/e2e/stubs"


def ssh(cmd: str, timeout=60):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST, cmd], capture_output=True,
                          text=True, timeout=timeout)


def enabled() -> bool:
    if os.environ.get("NS_E2E_REMOTE") != "1":
        return False
    try:
        return ssh("true", 15).returncode == 0
    except subprocess.SubprocessError:
        return False


def remote_json(name: str, rel: str):
    r = ssh(f"cat {RROOT}/runs/{name}/{rel}")
    return json.loads(r.stdout) if r.returncode == 0 else None


def wait_remote(name: str, pred, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = remote_json(name, "status.json")
        if st and pred(st):
            return st
        time.sleep(1.0)
    return None


@unittest.skipUnless(enabled(), "set NS_E2E_REMOTE=1 and make sure ssh to tyrone works")
class Remote(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="rhremote"))
        cls.env = dict(os.environ, NS_LOCAL_RUNS=str(cls.tmp / "runs"))
        # a small frames folder: 6 of the honka frames
        cls.frames = cls.tmp / "frames"
        cls.frames.mkdir()
        for f in sorted(FRAMES.glob("*.jpg"))[:6]:
            shutil.copy2(f, cls.frames / f.name)

    @classmethod
    def tearDownClass(cls):
        ssh(f"rm -rf {RROOT}/runs/rh-e2e-*", 60)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cfg(self, obj) -> str:
        p = self.tmp / f"cfg{abs(hash(json.dumps(obj, sort_keys=True)))}.json"
        p.write_text(json.dumps(obj))
        return str(p)

    def ns(self, *args, timeout=240, **kw):
        return subprocess.run([sys.executable, NS, *args], capture_output=True, text=True, timeout=timeout, env=self.env, **kw)

    # ---- 0.5
    def test_0_5_doctor_reports_everything_within_60s(self):
        t0 = time.time()
        r = self.ns("doctor", timeout=120)
        dt = time.time() - t0
        out = r.stdout
        for needle in ("ssh", "disk free (tyrone)", "GPU memory free", "namma-space-gpu:0.1", "sha256:", "ffmpeg", "node",
                       "upload", "download", "disk free (Mac)"):
            self.assertIn(needle, out, out)
        self.assertLess(dt, 60, f"doctor took {dt:.0f}s")
        self.assertIn(r.returncode, (0, 1))

    # ---- 0.2
    def test_0_2_ssh_drop_keeps_remote_job_running_and_pull_recovers(self):
        name = "rh-e2e-drop"
        p = subprocess.Popen([sys.executable, NS, "run", "--frames", str(self.frames), "--name", name, "--no-check",
                              "--stub-cfg", self.cfg({"train": {"sleep": 12}}), "--runner-flag", STUB_FLAG],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=self.env,
                             start_new_session=True)
        st = wait_remote(name, lambda s: (s.get("stages", {}).get("train", {}).get("state") == "running"), 120)
        self.assertIsNotNone(st, "train never reached running")
        pid = st["pid"]
        # drop everything on the Mac side: ns and its ssh/rsync children
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        alive = ssh(f"kill -0 {pid} && echo alive")
        self.assertIn("alive", alive.stdout, "the remote runner died with the connection")
        st = wait_remote(name, lambda s: s.get("state") in ("done", "partial", "review", "failed", "stopped"), 90)
        self.assertIsNotNone(st)
        self.assertEqual((st["state"], st.get("exit")), ("done", 0))
        r = self.ns("pull", name, "--force")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        run = self.tmp / "runs" / name
        for rel in ("50_train/train.json", "70_grid/grid.json", "timings.json", "run.json", "logs/train.log"):
            self.assertTrue((run / rel).exists(), rel)

    # ---- 0.1 through ns
    def test_0_1_kill_runner_then_ns_run_from_train(self):
        name = "rh-e2e-resume"
        r = self.ns("run", "--frames", str(self.frames), "--name", name, "--no-check", "--detach",
                    "--stub-cfg", self.cfg({"train": {"sleep": 120}}), "--runner-flag", STUB_FLAG)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        st = wait_remote(name, lambda s: s.get("stages", {}).get("train", {}).get("state") == "running", 90)
        self.assertIsNotNone(st)
        before = {}
        for rel in ("10_frames/frames.json", "20_sfm/sfm.json", "40_align/align.json"):
            before[rel] = ssh(f"stat -c %Y.%y {RROOT}/runs/{name}/{rel}").stdout.strip()
            self.assertTrue(before[rel])
        ssh(f"kill {st['pid']}")  # exact PID, as the rules say
        st = wait_remote(name, lambda s: s.get("state") in ("stopped", "failed"), 60)
        self.assertIsNotNone(st)
        r = self.ns("run", "--name", name, "--from", "train", "--no-check", "--stub-cfg", self.cfg({}), "--runner-flag", STUB_FLAG)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for rel, v in before.items():
            self.assertEqual(ssh(f"stat -c %Y.%y {RROOT}/runs/{name}/{rel}").stdout.strip(), v, f"{rel} was rewritten")
        calls = ssh(f"cat {RROOT}/runs/{name}/stub_calls.log").stdout.splitlines()
        starts = lambda s: sum(1 for c in calls if c.startswith(s + " start"))
        self.assertEqual([starts("ingest"), starts("sfm"), starts("align")], [1, 1, 1])
        self.assertEqual(starts("train"), 2)
        self.assertTrue(remote_json(name, "50_train/train.json"))

    # ---- 0.4
    def test_0_4_two_runs_at_once_do_not_collide(self):
        cfg = self.cfg({"train": {"sleep": 4}})
        for n in ("rh-e2e-a", "rh-e2e-b"):
            r = self.ns("run", "--frames", str(self.frames), "--name", n, "--no-check", "--detach", "--stub-cfg", cfg,
                        "--runner-flag", STUB_FLAG)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for n in ("rh-e2e-a", "rh-e2e-b"):
            st = wait_remote(n, lambda s: s.get("state") in ("done", "partial", "review", "failed", "stopped"), 120)
            self.assertIsNotNone(st, n)
            self.assertEqual(st.get("exit"), 0, n)
        a = remote_json("rh-e2e-a", "10_frames/frames.json")
        b = remote_json("rh-e2e-b", "10_frames/frames.json")
        self.assertEqual((a["n_kept"], b["n_kept"]), (6, 6))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg missing")
    def test_take_b_runs_as_a_second_run_in_parallel(self):
        v = self.tmp / "tiny.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-t", "1",
                        "-pix_fmt", "yuv420p", str(v)], check=True)
        r = self.ns("run", str(v), "--name", "rh-e2e-tb", "--take-b", str(v), "--no-check", "--no-pull", "--height", "170",
                    "--measure", "length=4.12", "--stub-cfg", self.cfg({}), "--runner-flag", STUB_FLAG)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for n in ("rh-e2e-tb", "rh-e2e-tb-b"):
            self.assertEqual(remote_json(n, "status.json")["exit"], 0, n)
            meta = remote_json(n, "input/meta.json")
            self.assertEqual(meta["height_cm"], 170.0)
            self.assertEqual(meta["measures"][0], {"what": "length", "metres": 4.12, "method": "tape"})
        self.assertEqual(remote_json("rh-e2e-tb-b", "input/meta.json")["takes"][0]["id"], "B")
        self.assertIn("tiny.mp4", ssh(f"ls {RROOT}/runs/rh-e2e-tb-b/input").stdout)

    # ---- a runner killed with -9 must not leave `ns run` polling forever
    def test_dead_runner_is_detected(self):
        name = "rh-e2e-dead"
        p = subprocess.Popen([sys.executable, NS, "run", "--frames", str(self.frames), "--name", name, "--no-check",
                              "--no-pull", "--stub-cfg", self.cfg({"train": {"sleep": 120}}), "--runner-flag", STUB_FLAG],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=self.env)
        st = wait_remote(name, lambda s: s.get("stages", {}).get("train", {}).get("child_pid"), 120)
        self.assertIsNotNone(st)
        ssh(f"kill -9 {st['pid']} {st['stages']['train']['child_pid']}")  # exact PIDs only
        t0 = time.time()
        out, _ = p.communicate(timeout=120)
        self.assertEqual(p.returncode, 20, out)
        self.assertIn("is gone without a final status", out)
        self.assertLess(time.time() - t0, 90)

    # ---- 0.8
    def test_0_8_rescale_changes_t_final_and_grid_without_retraining(self):
        name = "rh-e2e-rescale"
        r = self.ns("run", "--frames", str(self.frames), "--name", name, "--no-check", "--stub-cfg", self.cfg({}),
                    "--runner-flag", STUB_FLAG)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        grid0 = remote_json(name, "70_grid/grid.json")
        t_train = ssh(f"stat -c %Y.%y {RROOT}/runs/{name}/50_train/train.json").stdout.strip()
        t0 = time.time()
        r = self.ns("rescale", name, "--factor", "2", "--force", "--runner-flag", STUB_FLAG)
        dt = time.time() - t0
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertLess(dt, 300, f"rescale took {dt:.0f}s")
        exp = remote_json(name, "60_export/export.json")
        grid1 = remote_json(name, "70_grid/grid.json")
        self.assertEqual(exp["T_final"][0][0], 2.0)
        self.assertAlmostEqual(grid1["room_extent_m"][0], 2 * grid0["room_extent_m"][0])
        self.assertEqual(ssh(f"stat -c %Y.%y {RROOT}/runs/{name}/50_train/train.json").stdout.strip(), t_train)
        calls = ssh(f"cat {RROOT}/runs/{name}/stub_calls.log").stdout.splitlines()
        self.assertEqual(sum(1 for c in calls if c.startswith("train start")), 1)
        self.assertEqual(sum(1 for c in calls if c.startswith("grid start")), 2)
        local = self.tmp / "runs" / name
        self.assertEqual(json.loads((local / "70_grid" / "grid.json").read_text())["room_extent_m"], grid1["room_extent_m"])

    @unittest.skipUnless(os.environ.get("NS_E2E_DOCKER") == "1", "set NS_E2E_DOCKER=1 for the real-container run")
    def test_stub_stages_in_real_containers(self):
        name = "rh-e2e-docker"
        r = self.ns("run", "--frames", str(self.frames), "--name", name, "--no-check", "--quick",
                    "--runner-flag", "--stub-dir test/e2e/stubs", timeout=1500)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rj = remote_json(name, "run.json")
        self.assertTrue(all(v.startswith("sha256:") for v in rj["images"].values()), rj["images"])


if __name__ == "__main__":
    unittest.main()
