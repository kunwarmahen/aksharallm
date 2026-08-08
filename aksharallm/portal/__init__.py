"""A tiny local web portal for driving and watching training runs.

`aksharallm.portal.runs` is the model (what a run is, what state it's in, how to start and
stop it); `aksharallm.portal.server` is the view (a stdlib HTTP server and one page).
`aksharallm.portal.explain` is the second half of the page: a source browser that hands the
lines you highlight to a local Ollama model and streams back an explanation of them.

It deliberately adds **no dependency**: `http.server` from the standard library, and
hand-written SVG charts in the browser. The portal is a thin skin over the same
`scripts/phase2.sh` and `scripts/stop.sh` you would run by hand, so there is exactly one
code path for starting and stopping a run.

Read with: docs/10-running-and-watching.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from .explain import ExplainConfig, SourceTree
from .runs import RunStore, repo_root

__all__ = ["ExplainConfig", "RunStore", "SourceTree", "repo_root"]
