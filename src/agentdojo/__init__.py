"""Compatibility shim for older absolute imports.

The main package is `odile`, but a large amount of preserved research code still
imports `agentdojo.*`. Expose the `odile` package path under the old name so the
ported code can run without duplicating the full tree.
"""

from odile import *  # noqa: F401,F403
from odile import __path__ as _odile_path

__path__ = _odile_path
