from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools import builtin


def test_normalize_adds_https_when_missing_scheme():
    assert builtin._normalize_url("example.com") == "https://example.com"
    assert builtin._normalize_url("  example.com ") == "https://example.com"


def test_normalize_preserves_supported_schemes():
    assert builtin._normalize_url("http://example.com") == "http://example.com"
    assert builtin._normalize_url("https://example.com") == "https://example.com"
    assert builtin._normalize_url("file:///tmp/data.txt") == "file:///tmp/data.txt"
