"""Population snapshot failures must propagate instead of restarting forever."""

from multiprocessing import Pool

import pytest

from population_analysis import _init_render_worker, _render_and_save


@pytest.mark.parametrize("corrupt", [False, True], ids=["missing", "corrupt"])
def test_snapshot_load_failure_returns_to_parent(tmp_path, corrupt):
    snapshot = tmp_path / "population.pkl"
    if corrupt:
        snapshot.write_bytes(b"incomplete snapshot")

    # No plot should run when the snapshot could not be loaded. The bounded
    # wait catches the worker-respawn hang that an initializer exception causes.
    with Pool(1, initializer=_init_render_worker, initargs=(snapshot,)) as pool:
        result = pool.map_async(_render_and_save, [(None, tmp_path / "figure.svg", {})])
        with pytest.raises(RuntimeError, match="snapshot load error"):
            result.get(timeout=15)

    assert not (tmp_path / "figure.svg").exists()
