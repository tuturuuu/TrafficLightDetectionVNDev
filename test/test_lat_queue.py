"""Exercise real Linux handoff without stopping the running training process."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/runtime/lat_visdrone_queue.py'
spec = importlib.util.spec_from_file_location('lat_queue', SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


@pytest.fixture
def stopped_controller(tmp_path):
    processes = []

    def create(exit_code):
        # Parent would advance to the next queue item if it were allowed to run.
        ready = tmp_path / 'child_ready'
        child_script = tmp_path / 'child.py'
        child_script.write_text(f'import time, sys\nfrom pathlib import Path\n'
                                f'Path({str(ready)!r}).touch()\ntime.sleep(0.3)\nsys.exit({exit_code})\n')
        parent = subprocess.Popen(
            ['bash', '-c', f'{shlex.quote(sys.executable)} {shlex.quote(str(child_script))} & child=$!; '
             'echo "$child"; wait "$child"; echo NEXT_STAGE'],
            stdout=subprocess.PIPE, text=True, start_new_session=True)
        processes.append(parent)
        child = int(parent.stdout.readline())
        os.kill(parent.pid, signal.SIGSTOP)
        deadline = time.monotonic() + 2
        while queue.process_info(parent.pid)['state'] != 'T':
            assert time.monotonic() < deadline
            time.sleep(.005)
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(.005)
        manifest = {'parent': queue.process_info(parent.pid), 'child': queue.process_info(child)}
        path = tmp_path / 'adoption.json'
        queue.atomic_json(path, manifest)
        return parent, child, path

    yield create
    for parent in processes:
        if parent.poll() is None:
            os.kill(parent.pid, signal.SIGCONT)
        parent.communicate(timeout=5)


@pytest.mark.parametrize('exit_code', [0, 7])
def test_handoff_preserves_child_and_reads_actual_exit(tmp_path, stopped_controller, exit_code):
    parent, child, manifest = stopped_controller(exit_code)
    assert queue.process_info(child)['state'] != 'T'
    if exit_code:
        with pytest.raises(RuntimeError, match='exit code 7'):
            queue.adopt_detector(tmp_path, 1, manifest, poll_seconds=.01)
    else:
        queue.adopt_detector(tmp_path, 1, manifest, poll_seconds=.01)
    assert json.loads((tmp_path / 'adopted_detector_exit.json').read_text())['exit_code'] == exit_code
    output, _ = parent.communicate(timeout=5)
    assert parent.returncode == -signal.SIGKILL
    assert 'NEXT_STAGE' not in output


def test_handoff_rejects_pid_identity_mismatch(tmp_path, stopped_controller):
    parent, _, path = stopped_controller(0)
    manifest = json.loads(path.read_text())
    manifest['child']['start_ticks'] += 1
    queue.atomic_json(path, manifest)
    with pytest.raises(RuntimeError, match='identity changed'):
        queue.adopt_detector(tmp_path, 1, path, poll_seconds=.01)
    assert queue.process_info(parent.pid)['state'] == 'T'
    assert not (tmp_path / 'handoff_ready.json').exists()


def test_seed_lock_rejects_concurrent_owner(tmp_path):
    path = tmp_path / '.seed100.lock'
    with queue.lock(path, nonblocking=True):
        with pytest.raises(BlockingIOError):
            with queue.lock(path, nonblocking=True):
                pytest.fail('Concurrent worker acquired the same seed')


def test_aggregation_waits_for_every_seed(tmp_path):
    queue.atomic_json(tmp_path / 'seed100.completed.json', {'seed': 100})
    assert queue.maybe_aggregate(tmp_path) is False
    assert not (tmp_path / 'all_seeds.completed.json').exists()
