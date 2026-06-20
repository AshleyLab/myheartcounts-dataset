"""Ensure these tests import from this repo, not a sibling editable install.

The dev VM has both this repo and the upstream private repo installed in
editable mode; both add their ``src/`` to ``sys.path``. The order is
filesystem-dependent and the private one happens to win, which would silently
test the wrong ``imputation_evaluation`` package. This conftest prepends this
repo's ``src/`` so the public module under test always wins.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
_REPO_SRC_STR = str(_REPO_SRC)
if _REPO_SRC_STR in sys.path:
    sys.path.remove(_REPO_SRC_STR)
sys.path.insert(0, _REPO_SRC_STR)

# Evict any submodule of imputation_evaluation that may have been imported
# from the wrong location before this conftest ran.
for _mod_name in list(sys.modules):
    if _mod_name == "imputation_evaluation" or _mod_name.startswith("imputation_evaluation."):
        del sys.modules[_mod_name]
