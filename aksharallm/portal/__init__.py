"""A tiny local web portal for driving and watching training runs.

`aksharallm.portal.runs` is the model (what a run is, what state it's in, how to start and
stop it); `aksharallm.portal.server` is the view (a stdlib HTTP server and one page).

It deliberately adds **no dependency**: `http.server` from the standard library, and
hand-written SVG charts in the browser. The portal is a thin skin over the same
`scripts/phase2.sh` and `scripts/stop.sh` you would run by hand, so there is exactly one
code path for starting and stopping a run.
"""

from .runs import RunStore, repo_root

__all__ = ["RunStore", "repo_root"]
