"""Installation planning is safe, repeatable, and independent of shell activation."""

from pathlib import Path

import pytest


def test_cpu_install_uses_external_environment_and_constraints(tmp_path):
    from scripts.install_researcher import build_install_plan

    root = tmp_path / "research project"
    environment = tmp_path / "environment"
    plan = build_install_plan(root, environment, "/usr/bin/python3", uv="/bin/uv")
    install = plan.commands[-1]
    assert "--torch-backend" in install and "cpu" in install
    assert "--constraint" in install
    assert str(root / "requirements-researcher.txt") in install
    assert f"{root}[researcher]" in install
    assert not any(
        "ollama" in argument or "sudo" in argument
        for command in plan.commands
        for argument in command
    )


def test_pip_fallback_installs_cpu_torch_before_researcher_package(tmp_path):
    from scripts.install_researcher import build_install_plan

    root = tmp_path / "project"
    root.mkdir()
    (root / "requirements-researcher.txt").write_text("torch==2.15.0\n")
    plan = build_install_plan(root, tmp_path / "env", "/bin/python", uv=None)
    assert any(
        "https://download.pytorch.org/whl/cpu" in command for command in plan.commands
    )
    assert "--constraint" in plan.commands[-1]


def test_environment_inside_repository_is_rejected(tmp_path):
    from scripts.install_researcher import build_install_plan

    with pytest.raises(ValueError, match="outside"):
        build_install_plan(tmp_path, tmp_path / "venv", "/bin/python")


def test_launch_command_is_local_browser_app_without_activation(tmp_path):
    from scripts.install_researcher import launch_command

    command = launch_command(tmp_path / "project", tmp_path / "environment", 8502)
    assert "streamlit" in command
    assert "--server.address=127.0.0.1" in command
    assert "--server.port=8502" in command
    assert "activate" not in " ".join(command)


def test_package_and_runtime_versions_agree():
    import tomllib

    from eeg_rag.__version__ import __version__

    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]
    assert __version__ == project["version"]
