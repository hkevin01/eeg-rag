#!/usr/bin/env python3
"""Install and launch EEG-RAG for researchers without shell activation or Docker."""

import argparse
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class InstallPlan:
    """Inspectable commands for an isolated CPU-first researcher installation."""

    commands: list[list[str]]
    environment: Path


def environment_python(environment: Path) -> Path:
    """Locate the interpreter in a Windows or POSIX virtual environment."""
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def build_install_plan(
    root: Path, environment: Path, python: str, uv: Optional[str] = None,
) -> InstallPlan:
    """Build installation commands without executing them or altering files.

    Args:
        root: Source checkout containing package metadata and locked constraints.
        environment: Environment location outside the source checkout.
        python: Interpreter to use when creating the environment.
        uv: Optional installed uv command for fast CPU-aware installation.

    Returns:
        Ordered creation and installation commands.

    Raises:
        ValueError: The environment is inside the repository or constraints lack Torch.
    """
    root = root.resolve()
    environment = environment.resolve()
    if environment == root or root in environment.parents:
        raise ValueError("The environment must be outside the repository")
    interpreter = str(environment_python(environment))
    constraints = root / "requirements-researcher.txt"
    commands = []
    if uv:
        if not environment_python(environment).exists():
            commands.append([uv, "venv", str(environment), "--python", python, "--seed"])
        commands.append([
            uv, "pip", "install", "--python", interpreter, "--torch-backend", "cpu",
            "--constraint", str(constraints), "--editable", f"{root}[researcher]",
        ])
    else:
        if not environment_python(environment).exists():
            commands.append([python, "-m", "venv", str(environment)])
        match = re.search(r"^torch==([^\s;]+)", constraints.read_text(), re.MULTILINE)
        if not match:
            raise ValueError("Researcher constraints must pin torch")
        commands.extend([
            [interpreter, "-m", "pip", "install", f"torch=={match.group(1)}",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            [interpreter, "-m", "pip", "install", "--constraint", str(constraints),
             "--editable", f"{root}[researcher]"],
        ])
    return InstallPlan(commands, environment)


def launch_command(root: Path, environment: Path, port: int = 8501) -> list[str]:
    """Build a loopback-only browser UI command using the installed interpreter.

    Args:
        root: EEG-RAG checkout.
        environment: Researcher environment, requiring no activation.
        port: Local port; change it when another application is already listening.

    Returns:
        Streamlit command with local-only binding and browser startup enabled.

    Raises:
        ValueError: Port is outside the valid TCP range.
    """
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    return [str(environment_python(environment)), "-m", "streamlit", "run",
            str(root / "src/eeg_rag/web_ui/app.py"), "--server.address=127.0.0.1",
            f"--server.port={port}", "--server.headless=false"]


def main() -> None:
    """Install, check, or launch a checkout with visible progress and clear errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--launch", action="store_true", help="Open the installed browser interface")
    action.add_argument("--check", action="store_true", help="Check installation and show package versions")
    action.add_argument("--plan", action="store_true", help="Preview commands without changing anything")
    parser.add_argument("--environment", type=Path, help="Optional environment location outside the checkout")
    parser.add_argument("--port", type=int, default=8501, help="Browser interface port (default: 8501)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    identity = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local/share"))
    environment = args.environment or base / "eeg-rag" / f"researcher-{identity}"
    try:
        if not (3, 11) <= sys.version_info[:2] < (3, 15):
            raise ValueError("Please use Python 3.11-3.14 (Python 3.12 recommended)")
        plan = build_install_plan(root, environment, sys.executable, shutil.which("uv"))
        if args.plan:
            for command in plan.commands:
                print(shlex.join(command))
            return
        if args.launch or args.check:
            if not environment_python(environment).exists():
                raise ValueError("Run this script without options once to install EEG-RAG")
        else:
            print("Installing EEG-RAG for EEG research. No administrator access is needed.")
            print("Internet access and several GB of free disk space are required.")
            for index, command in enumerate(plan.commands, 1):
                print(f"Step {index}/{len(plan.commands)}: installing isolated dependencies", flush=True)
                subprocess.run(command, cwd=root, check=True)
            for directory in ("data/raw", "data/processed", "data/embeddings", "logs"):
                (root / directory).mkdir(parents=True, exist_ok=True)
            if not (root / ".env").exists() and (root / ".env.example").exists():
                with (root / ".env").open("x") as config:
                    config.write((root / ".env.example").read_text())
                print("Created .env from the example; existing configurations are never replaced.")
        if args.launch:
            print(f"Opening EEG-RAG at http://127.0.0.1:{args.port}", flush=True)
            subprocess.run(launch_command(root, environment, args.port), cwd=root, check=True)
            return
        subprocess.run([
            str(environment_python(environment)), "-c",
            "from importlib.metadata import version; import streamlit, eeg_rag; "
            "print('EEG-RAG', version('eeg-rag')); "
            "print('Sentence Transformers', version('sentence-transformers')); "
            "print('Qdrant client', version('qdrant-client')); print('Browser interface ready')",
        ], cwd=root, check=True)
        print("Next: python scripts/install_researcher.py --launch")
        print("Literature search does not require an LLM key; generated answers need a configured provider.")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Setup could not finish: {error}", file=sys.stderr)
        print("Check internet access and free disk space, then rerun the same command.", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
