"""Shared, explicit protocol for the November 2026 Talbot rerun."""
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import threading
import time

import numpy as np

from lsi.config import SystemConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOCK = threading.RLock()
REFERENCE_COMMIT = "71ab81b18c9f017c6ddd3dfc302d18b0a2ba2542"


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    with LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                        allow_nan=False) + "\n")
        temporary.replace(path)


def digest(array):
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def event(name, **values):
    row = {**values, "timestamp_utc": utc(), "event": name}
    with LOCK:
        RESULTS.mkdir(parents=True, exist_ok=True)
        with (RESULTS / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


@dataclass(frozen=True)
class TalbotConfig(SystemConfig):
    source_kind: str = "multi64"

    @property
    def sources(self):
        if self.source_kind == "single":
            return ((7, 7),)
        if self.source_kind != "multi64":
            raise ValueError(self.source_kind)
        return tuple((i, j) for i in range(3, 11) for j in range(3, 11))


def config(kind="multi64"):
    raw = json.loads((ROOT / "reference" / "config_file.json").read_text())
    names = {field.name for field in fields(SystemConfig)}
    result = TalbotConfig(**{key: raw[key] for key in names}, source_kind=kind)
    assert result.n == 1680 and result.n * result.up_sample_grating == 3360
    assert len(result.sources) == (1 if kind == "single" else 64)
    return result


def truth_for(model):
    result = np.zeros(35)
    result[np.array([7, 25, 11, 15]) - 1] = [.25, .15, .05, .2]
    wavefront = model.wavefront(result)[model.pupil]
    result *= .05 / np.sqrt(np.mean(wavefront * wavefront))
    return result


class Monitor:
    def __init__(self, stage, kind=None):
        self.started = time.monotonic()
        self.state = dict(status="running", stage=stage, source_kind=kind,
                          eta_remaining_seconds=None)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def update(self, **values):
        import psutil
        with LOCK:
            self.state.update(values, timestamp_utc=utc(),
                              elapsed_seconds=time.monotonic() - self.started,
                              rss_gib=psutil.Process().memory_info().rss / 1024**3)
            save(RESULTS / "status.json", self.state)

    def _loop(self):
        while not self.stop.wait(30):
            self.update()
            with LOCK:
                with (RESULTS / "eta.jsonl").open("a") as stream:
                    stream.write(json.dumps(self.state, allow_nan=False) + "\n")
            event("heartbeat", **self.state)
            if self.state["rss_gib"] > 12:
                import os
                event("memory_limit", rss_gib=self.state["rss_gib"])
                os._exit(74)

    def __enter__(self):
        self.update()
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.update(status="failed" if kind else "complete",
                    error=repr(value) if kind else None,
                    eta_remaining_seconds=0 if not kind else None)
