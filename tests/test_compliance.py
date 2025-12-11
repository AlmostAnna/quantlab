# tests/test_compliance.py
"""
Tests for license compliance checking.

This module contains tests to ensure proper attribution and license notices
are included for dependencies like QuantLib.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
NOTICES_PATH = REPO_ROOT / "NOTICES"
SRC_PATH = REPO_ROOT / "src"
TESTS_PATH = REPO_ROOT / "tests"


def test_notices_file_exists():
    """Test that the NOTICES file exists in the repository root."""
    assert NOTICES_PATH.exists(), "NOTICES file missing in repo root"


def test_quantlib_license_in_notices():
    """Test that QuantLib license information is present in NOTICES file."""
    content = NOTICES_PATH.read_text(encoding="utf-8")
    required_phrases = [
        "Copyright (c) 2000–2025 StatPro Italia srl",
        "Redistribution and use in source and binary forms",
        "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS",
    ]
    for phrase in required_phrases:
        assert phrase in content, f"Missing required license phrase: {phrase!r}"


def test_quantlib_usage_has_attribution():
    """Check that any .py file importing QuantLib includes an attribution comment."""
    ql_import_re = re.compile(r"^\s*(import|from)\s+QuantLib", re.MULTILINE)
    attribution_re = re.compile(
        r"QuantLib.*licensed under.*BSD|BSD.*license.*QuantLib", re.IGNORECASE
    )

    for py_file in list(SRC_PATH.rglob("*.py")) + list(TESTS_PATH.rglob("*.py")):
        code = py_file.read_text(encoding="utf-8")
        if ql_import_re.search(code):
            assert attribution_re.search(code), (
                f"File {py_file} imports QuantLib but lacks attribution comment. "
                "Add: '# This module uses QuantLib (...), licensed under the BSD 3-Clause License.'"  # noqa: E501
            )
