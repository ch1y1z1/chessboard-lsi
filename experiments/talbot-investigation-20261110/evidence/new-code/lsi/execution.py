"""CLI execution limits; these checks never construct a diffraction model."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import tempfile


def requires_large_run(config):
    """Only the tiny test-sized scene can run without explicit opt-in."""
    return (config.n > 80 or config.n * config.up_sample_grating > 160
            or len(config.sources) > 4)


@contextmanager
def exclusive_study(lock_path=None):
    """One study per local user, including across worktrees; OS releases on exit."""
    path = (Path(lock_path) if lock_path is not None else
            Path(tempfile.gettempdir()) / f'chessboard-lsi-study-{os.getuid()}.lock')
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another chessboard-lsi study is running; run studies sequentially') from error
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()) + '\n')
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def interrupt_on_termination():
    """Let the CLI record partial results when receiving SIGTERM."""
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('terminated by SIGTERM')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
