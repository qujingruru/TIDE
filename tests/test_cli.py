from __future__ import annotations

import json

from pytest import CaptureFixture

from tide.cli import main


def test_show_config_prints_resolved_json(capsys: CaptureFixture[str]) -> None:
    assert main(["show-config"]) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["seed"] == 42
    assert data["models"]["language_model"]["name"] == "Qwen/Qwen2.5-1.5B"
