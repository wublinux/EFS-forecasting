from adaptforecast.config import load_config
from adaptforecast.matlab_bridge import matlab_available


def test_matlab_source_directory_is_not_an_executable(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "matlab").mkdir()
    monkeypatch.setattr("adaptforecast.matlab_bridge.shutil.which", lambda _: None)
    assert matlab_available("matlab") is False


def test_matlab_executable_can_be_supplied_by_environment(monkeypatch) -> None:
    monkeypatch.setenv("ADAPTFORECAST_MATLAB_EXECUTABLE", "/opt/matlab/bin/matlab")
    config = load_config("configs/benchmark.smoke.yaml")
    assert config.efs.executable == "/opt/matlab/bin/matlab"
