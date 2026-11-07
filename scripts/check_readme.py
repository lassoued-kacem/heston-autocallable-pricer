"""Check README numbers against fresh script output.

Every README block between ``<!-- numbers-from: scripts/X.py -->`` and
``<!-- end-numbers -->`` may only contain numbers that appear in the output of a fresh run
of scripts/X.py. A "number" here is any token with a decimal point or a thousands
separator (integers such as step counts are not checked). Matching is exact, digit for
digit and including the sign; an explicit "+" counts as no sign. Exits 1 on any mismatch.

    .venv/bin/python scripts/check_readme.py
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = re.compile(r"(?<![\w.])-?\+?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+)")
BLOCK = re.compile(r"<!-- numbers-from: (\S+) -->(.*?)<!-- end-numbers -->", re.S)


def numbers(text):
    return {m.group(0).replace("+", "") for m in NUM.finditer(text)}


def main():
    readme = (ROOT / "README.md").read_text()
    blocks = BLOCK.findall(readme)
    if not blocks:
        print("no marked blocks found")
        return 1
    outputs, failed = {}, False
    for script, body in blocks:
        if script not in outputs:
            print(f"running {script} ...", flush=True)
            res = subprocess.run([sys.executable, str(ROOT / script)], capture_output=True,
                                 text=True, cwd=ROOT, check=True)
            outputs[script] = numbers(res.stdout)
        need = numbers(body)
        missing = sorted(need - outputs[script])
        status = "OK" if not missing else f"MISSING {missing}"
        print(f"  {script}: {len(need)} numbers checked, {status}")
        failed |= bool(missing)
    print("README numbers match script output" if not failed else "README has numbers not in script output")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
