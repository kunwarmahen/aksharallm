"""Long context: making a trained model read further than its window, and measuring it.

Four pieces, in the order you use them:

    extend.py     turn a 1k checkpoint into a 4k one (a config edit; no weights change)
    curve.py      loss by position -- is it still fluent out there?
    haystack.py   needle in a haystack -- can it still *retrieve* out there?
    __main__.py   the CLI that drives all three

The scaling maths itself lives one level down in `aksharallm/model/rope.py`, next to the
model it modifies, because that is the file a reader of doc 3 will already have open.

Nothing is re-exported here on purpose. `extend` and `haystack` are the natural names for
both a module and its main function, and binding the function at package level shadows the
module — so `from aksharallm.longctx import extend` would hand you a function where you
asked for a file, and the error surfaces somewhere unrelated. Import from the submodule:

    from aksharallm.longctx.extend import extend
    from aksharallm.longctx.curve import position_curve, cliff
    from aksharallm.longctx.haystack import run

Read with: docs/18-long-context.md -- the chapter this implements; it ends with the order to
read these files in.
"""
