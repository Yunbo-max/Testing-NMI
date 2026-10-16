"""Guard against unsupported README mathematics on constrained renderers."""
from pathlib import Path


def test_readme_math_uses_supported_macros():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert r"\operatorname" not in readme
    assert r"\mathrm{RoPE}" in readme
    assert r"\mathrm{RMSNorm}" in readme
    assert r"\mathbin{\|}" not in readme
    assert readme.count("$$") % 2 == 0


def test_compatibility_stated_as_unverified():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "not yet verified" in readme.lower()
    assert "Do not delete any" in readme
    assert "full 27B" in readme
