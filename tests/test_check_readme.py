"""Unit tests of the README number checker's parsing (the scripts themselves are not run)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_readme", Path(__file__).resolve().parents[1] / "scripts" / "check_readme.py")
CR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CR)


def test_numbers_keep_sign_and_ignore_plain_integers():
    got = CR.numbers("MC = -0.0801 +- 0.0088, +2.9462 at 48 steps, 1,000,000 paths, v0.04")
    assert got == {"-0.0801", "0.0088", "2.9462", "1,000,000"}


def test_sign_mismatch_is_detected():
    readme_block = "D = -0.0801"
    output = "D = +0.0801"
    assert CR.numbers(readme_block) - CR.numbers(output) == {"-0.0801"}


def test_blocks_are_parsed_with_their_script():
    text = "x\n<!-- numbers-from: scripts/a.py -->\n| 1.25 |\n<!-- end-numbers -->\ny 9.99"
    (script, body), = CR.BLOCK.findall(text)
    assert script == "scripts/a.py" and CR.numbers(body) == {"1.25"}


def test_block_may_name_several_scripts():
    text = "<!-- numbers-from: scripts/a.py scripts/b.py -->\n0.5\n<!-- end-numbers -->"
    (scripts, _), = CR.BLOCK.findall(text)
    assert scripts.split() == ["scripts/a.py", "scripts/b.py"]
