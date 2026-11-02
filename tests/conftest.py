"""Keep automated tests from opening interactive Qt dialogs."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox


@pytest.fixture(autouse=True)
def disable_interactive_qt_dialogs(monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.No
    )
    for method in ("critical", "information", "warning"):
        monkeypatch.setattr(
            QMessageBox, method, lambda *_args, **_kwargs: QMessageBox.Ok
        )
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *_args, **_kwargs: ("", "")
    )
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *_args, **_kwargs: ("", "")
    )
    monkeypatch.setattr(
        QInputDialog, "getInt", lambda *_args, **_kwargs: (0, False)
    )
