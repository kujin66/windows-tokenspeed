import hashlib
import importlib.metadata as md
import json
import os
import runpy
import sys
from pathlib import Path

import flashinfer

def main():
    root = Path(__file__).resolve().parents[1]
    report = Path(sys.argv.pop(1))
    script = sys.argv.pop(1)
    version = md.version('flashinfer-python')
    module = Path(flashinfer.__file__).resolve()
    if version != '0.6.17+g2ea3c24b5dac' or not module.is_relative_to(root/'.venv/flashinfer0617'):
        raise RuntimeError(f'Wrong FlashInfer import/version: {module}, {version}')
    report.write_text(json.dumps(dict(flashinfer_version=version,flashinfer_module=str(module),module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),backend=os.environ['TOKENSPEED_COMPARISON_OPS']),indent=2))
    sys.argv[0] = script
    runpy.run_path(script,run_name='__main__')


if __name__ == '__main__':
    main()
