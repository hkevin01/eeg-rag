"""Lightweight visual layout regression checks for Streamlit UI breakpoints."""

from __future__ import annotations

import re
import socket
import subprocess
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _wait_for_server(port: int, timeout_sec: float = 20.0) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for Streamlit server on port {port}")


def _assert_no_horizontal_overflow(page) -> None:
    has_horizontal_overflow = page.evaluate(
        """
        () => document.documentElement.scrollWidth > window.innerWidth + 1
        """
    )
    assert has_horizontal_overflow is False


def _assert_no_card_overlap(page) -> None:
    has_overlap = page.evaluate(
        """
        () => {
            const cards = Array.from(document.querySelectorAll('.wcag-card'));
            for (let i = 0; i < cards.length; i += 1) {
                const a = cards[i].getBoundingClientRect();
                for (let j = i + 1; j < cards.length; j += 1) {
                    const b = cards[j].getBoundingClientRect();
                    const overlap = !(a.right <= b.left || a.left >= b.right || a.bottom <= b.top || a.top >= b.bottom);
                    if (overlap) {
                        const verticalGap = Math.min(Math.abs(a.bottom - b.top), Math.abs(b.bottom - a.top));
                        if (verticalGap < 1 && Math.abs(a.left - b.left) < 1 && Math.abs(a.right - b.right) < 1) {
                            continue;
                        }
                        return true;
                    }
                }
            }
            return false;
        }
        """
    )
    assert has_overlap is False


def _capture_and_assert(page, screenshot_path: Path) -> None:
    _assert_no_horizontal_overflow(page)
    _assert_no_card_overlap(page)
    page.screenshot(path=str(screenshot_path), full_page=True)
    assert screenshot_path.exists()


@pytest.mark.parametrize(
    "width,height",
    [
        (320, 900),
        (768, 1024),
        (1280, 1024),
    ],
)
def test_app_modular_layout_no_overflow_or_overlap(width: int, height: int, tmp_path: Path) -> None:
    """Ensure app_modular layout is non-overlapping and non-overflowing at key widths."""
    app_path = Path("src/eeg_rag/web_ui/app_modular.py").resolve()
    port = _find_free_port()

    process = subprocess.Popen(
        [
            ".venv/bin/python",
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.headless=true",
            "--server.address=127.0.0.1",
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=Path.cwd(),
    )

    try:
        _wait_for_server(port)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(
                f"http://127.0.0.1:{port}/?ui_test_seed=1",
                wait_until="networkidle",
            )
            page.wait_for_timeout(1500)

            _capture_and_assert(
                page,
                tmp_path / f"app_modular_default_{width}px.png",
            )

            page.get_by_role("tab", name=re.compile("Results History")).click()
            page.wait_for_timeout(700)
            _capture_and_assert(
                page,
                tmp_path / f"app_modular_results_tab_{width}px.png",
            )

            expander = page.get_by_text("What EEG biomarkers predict seizure recurrence", exact=False)
            expander.first.click()
            page.wait_for_timeout(500)
            _capture_and_assert(
                page,
                tmp_path / f"app_modular_expanded_cards_{width}px.png",
            )

            page.get_by_role("tab", name=re.compile("Agent Pipeline")).click()
            page.wait_for_timeout(500)
            _capture_and_assert(
                page,
                tmp_path / f"app_modular_agent_pipeline_{width}px.png",
            )

            page.get_by_role("tab", name=re.compile("Results History")).click()
            page.wait_for_timeout(500)
            page.mouse.wheel(0, 1800)
            page.wait_for_timeout(400)
            _capture_and_assert(
                page,
                tmp_path / f"app_modular_long_citations_{width}px.png",
            )

            browser.close()
    finally:
        process.terminate()
        process.wait(timeout=10)
