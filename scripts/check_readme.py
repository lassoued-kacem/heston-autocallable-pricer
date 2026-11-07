"""Check README numbers against fresh script output.

Every README block between ``<!-- numbers-from: scripts/X.py [scripts/Y.py ...] -->`` and
``<!-- end-numbers -->`` may only contain numbers that appear in the output of a fresh run
of the named script(s). A "number" here is any token with a decimal point or a thousands
separator (integers such as step counts are not checked). Matching is exact, digit for
digit and including the sign; an explicit "+" counts as no sign. Exits 1 on any mismatch.

convergence.py is run with --figures-dir pointing to a temporary directory, so the check
never rewrites docs/figures; the fresh PNGs are compared byte for byte with the committed
ones and the result is reported (a mismatch is informational: PNG bytes depend on the
matplotlib version, the numbers do not).

Limits (what this cannot catch): matching is by set membership, so a number placed in the
wrong row or label passes if it appears anywhere in the output, and a sign flip passes if
the other sign also appears. Plain integers, scientific notation, leading-dot decimals and
qualitative wording ("about half") are not checked, nor is anything outside the markers.
It is not run by pytest or the Stop hook (the scripts take about 90 s).

    .venv/bin/python scripts/check_readme.py
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = re.compile(r"(?<![\w.])-?\+?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+)")
BLOCK = re.compile(r"<!-- numbers-from: ([^>]+?) -->(.*?)<!-- end-numbers -->", re.S)
FIGURES = ("bias_vs_steps.png", "se_vs_paths.png")


def numbers(text):
    return {m.group(0).replace("+", "") for m in NUM.finditer(text)}


def main():
    readme = (ROOT / "README.md").read_text()
    blocks = BLOCK.findall(readme)
    if not blocks:
        print("no marked blocks found")
        return 1
    outputs, failed = {}, False
    with tempfile.TemporaryDirectory() as tmp:
        for scripts, body in blocks:
            have = set()
            for script in scripts.split():
                if script not in outputs:
                    print(f"running {script} ...", flush=True)
                    extra = ["--figures-dir", tmp] if script.endswith("convergence.py") else []
                    res = subprocess.run([sys.executable, str(ROOT / script), *extra],
                                         capture_output=True, text=True, cwd=ROOT, check=True)
                    outputs[script] = numbers(res.stdout)
                have |= outputs[script]
            need = numbers(body)
            missing = sorted(need - have)
            status = "OK" if not missing else f"MISSING {missing}"
            print(f"  {scripts}: {len(need)} numbers checked, {status}")
            failed |= bool(missing)
        for name in FIGURES:
            fresh, committed = Path(tmp) / name, ROOT / "docs/figures" / name
            if fresh.exists():
                same = committed.exists() and fresh.read_bytes() == committed.read_bytes()
                print(f"  figure {name}: {'identical to committed' if same else 'DIFFERS from committed (informational)'}")
    print("README numbers match script output" if not failed else "README has numbers not in script output")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
