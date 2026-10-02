"""Pre-flight for a recogniser run: does every corpus the config names exist? (docs/23)

Called by `scripts/audio.sh` for an `asr:` config. A recogniser reads `data.train` (a list)
and `data.val`, where a codec reads one `data.corpus`, so it needs its own check — and a
missing corpus should say which command builds it rather than fail inside the trainer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main(cfg_path: str) -> int:
    d = (yaml.safe_load(Path(cfg_path).read_text()) or {}).get("data") or {}
    train = d.get("train") or []
    paths = ([train] if isinstance(train, str) else list(train)) + ([d["val"]] if d.get("val") else [])
    bad = []
    for p in paths:
        f = Path(p) / "audio.bin"
        if f.is_file():
            print(f"    {f}: {f.stat().st_size / 2 / 16000 / 3600:.2f} h")
        else:
            bad.append(p)
    for p in bad:
        print(f"    ERROR: {p}/audio.bin does not exist.", file=sys.stderr)
        if p.startswith("data/audio/synth"):
            print(f"           build it:  .venv/bin/python -m aksharallm.audio corpus --out {p} --clips 400",
                  file=sys.stderr)
        else:
            name = Path(p).name
            print(f"           fetch it:  .venv/bin/python -m aksharallm.asr fetch {name}", file=sys.stderr)
            print(f"           pack it:   .venv/bin/python -m aksharallm.asr pack {name}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
