"""The Pictures tab's back end: the picture tokenizer's reports, its gallery, and two jobs.

* **reports** — every `pictures report` on disk (`logs/pictures/report-*.json`): the ladder
  (PSNR and what the reader sees, at each count of codebooks), the held-out combination,
  codebook use. Read beside the originals' row, the reader's positive control.
* **gallery** — the PNG rows the report wrote (original, then each codebook count), sent as
  data URIs with the report so the page needs no file route and can never be pointed at a
  file outside `logs/pictures/`.
* **compare** — two reports, paired picture by picture (`measure.compare`).
* **jobs** — `pictures report` (on the CPU) and `pictures read`, detached through `CliJobs`,
  the command printed. Training is the dashboard's Start (`imgtok-shapes` is in `LAUNCHERS`).
* **G1, a sentence to a picture** — `draw` runs inline on the CPU (128 sampling steps of a
  ~5M transformer, a second or two) and returns each picture with what the reader saw in it;
  `gen-*.json` reports (all 60 captions, seen and held-out) list, open and pair like the
  tokenizer's; `pictures codes` and `pictures gen-report` are jobs.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import torch

from ..audio.ttsreports import finite
from .jobs import CliJobs, JobError

RESULTS = Path("logs/pictures")
_NAME = re.compile(r"^(report|gen)-[A-Za-z0-9_.-]+\.json$")


class PicturesError(RuntimeError):
    pass


class Pictures:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.jobs = CliJobs(self.root, "logs/pictures/jobs", "aksharallm.pictures")

    def checkpoints(self) -> list[dict]:
        """Picture tokenizers, recognised by the `stage` their trainer writes."""
        out = []
        for path in sorted(self.root.glob("checkpoints/*/ckpt_*.pt")):
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except Exception:
                continue
            if blob.get("stage") != "picture":
                continue
            pc = blob.get("picture") or {}
            grid = pc.get("image_size", 64) // (2 ** len(pc.get("channels") or [0, 0, 0]))
            out.append({"rel": str(path.relative_to(self.root)), "step": blob.get("step"),
                        "grid": grid, "n_codebooks": pc.get("n_codebooks"),
                        "codebook_size": pc.get("codebook_size"),
                        "mtime": path.stat().st_mtime})
        return sorted(out, key=lambda r: r["mtime"], reverse=True)

    def generators(self) -> list[dict]:
        """G1 caption-to-picture checkpoints (`stage == "picgen"`)."""
        out = []
        for path in sorted(self.root.glob("checkpoints/*/ckpt_*.pt")):
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except Exception:
                continue
            if blob.get("stage") != "picgen":
                continue
            pg = blob.get("picgen") or {}
            out.append({"rel": str(path.relative_to(self.root)), "step": blob.get("step"),
                        "books": pg.get("books"), "tokenizer": pg.get("tokenizer"),
                        "best_val": blob.get("best_val"), "mtime": path.stat().st_mtime})
        return sorted(out, key=lambda r: r["mtime"], reverse=True)

    def gen_reports(self) -> list[dict]:
        out = []
        for p in sorted((self.root / RESULTS).glob("gen-*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                d = json.loads(p.read_text())
                out.append({"name": p.name, "run": d.get("run"), "step": d.get("step"),
                            "k": d.get("k"), "guidance": d.get("guidance"),
                            "seen_all": d["seen"]["all"],
                            "held_out_all": (d.get("held_out") or {}).get("all"),
                            "copies": d["seen"].get("copies"), "variety": d.get("variety")})
            except (OSError, ValueError, KeyError):
                continue
        return out

    def reports(self) -> list[dict]:
        folder = self.root / RESULTS
        out = []
        for p in sorted(folder.glob("report-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                d = json.loads(p.read_text())
                top = d["ladder"][-1]
                held = ((d.get("holdout") or {}).get("ladder") or [{}])[-1].get("all")
                out.append({"name": p.name, "run": d.get("run"), "step": d.get("step"),
                            "grid": d.get("grid"), "n_codebooks": d.get("n_codebooks"),
                            "n": d.get("n"), "psnr": top["psnr"], "all": top["all"],
                            "holdout_all": held})
            except (OSError, ValueError, KeyError, IndexError):
                continue
        return out

    def corpus(self) -> dict:
        m = self.root / "data/vision/shapes/manifest.json"
        if not m.is_file():
            return {"ready": False}
        try:
            n = len(json.loads(m.read_text()).get("captions") or [])
        except (OSError, ValueError):
            n = None
        return {"ready": True, "dir": "data/vision/shapes", "pictures": n,
                "holdout": (self.root / "data/vision/shapes-holdout/manifest.json").is_file()}

    def overview(self) -> dict:
        codes = sorted(str(p.parent.relative_to(self.root))
                       for p in self.root.glob("data/pictures/*/meta.json"))
        return finite({"checkpoints": self.checkpoints(), "reports": self.reports(),
                       "generators": self.generators(), "gen_reports": self.gen_reports(),
                       "codes": codes, "corpus": self.corpus(), "jobs": self.jobs.status()})

    def _load(self, name: str) -> dict:
        if not _NAME.match(name or ""):
            raise PicturesError(f"not a picture report: {name!r}")
        path = self.root / RESULTS / name
        if not path.is_file():
            raise PicturesError(f"no such report: {name}")
        return json.loads(path.read_text())

    def report(self, name: str) -> dict:
        """One report, with its gallery PNGs inlined. Each gallery path is re-checked to sit
        inside `logs/pictures/` before it is read, whatever the JSON says."""
        d = self._load(name)
        base = (self.root / RESULTS).resolve()
        pics = []
        for rel in d.get("gallery") or []:
            p = (base / rel).resolve()
            if base not in p.parents or p.suffix != ".png" or not p.is_file():
                continue
            pics.append("data:image/png;base64," + base64.b64encode(p.read_bytes()).decode())
        d["gallery_png"] = pics
        return finite(d)

    def compare(self, a: str, b: str) -> dict:
        from ..pictures.measure import compare
        x, y = self._load(a), self._load(b)
        if a.startswith("gen-") or b.startswith("gen-"):
            return self._gen_compare(a, b, x, y)
        out = {"a": a, "b": b, "rows": []}
        try:
            out["rows"].append({"set": "val pictures", **compare(x, y)})
            if x.get("holdout") and y.get("holdout"):
                out["rows"].append({"set": "held-out combination",
                                    **compare(x["holdout"], y["holdout"])})
        except ValueError as e:
            raise PicturesError(str(e)) from e
        out["command"] = f"python -m aksharallm.pictures compare {RESULTS}/{a} {RESULTS}/{b}"
        return finite(out)

    def _gen_compare(self, a: str, b: str, x: dict, y: dict) -> dict:
        from ..pictures.score import compare
        if not (a.startswith("gen-") and b.startswith("gen-")):
            raise PicturesError("pair two G1 reports, or two tokenizer reports, not one of each")
        try:
            parts = compare(x, y)
        except ValueError as e:
            raise PicturesError(str(e)) from e
        names = {"seen": "seen captions", "held_out": "held-out captions (purple triangles)"}
        return finite({"a": a, "b": b,
                       "rows": [{"set": names[k], **v} for k, v in parts.items()],
                       "command": f"python -m aksharallm.pictures gen-compare "
                                  f"{RESULTS}/{a} {RESULTS}/{b}"})

    def draw(self, spec: dict) -> dict:
        """Draw pictures for one caption, inline on the CPU, each read back by the reader.
        The model stays loaded between calls (keyed by checkpoint), like the Interp tab's."""
        import io

        from PIL import Image

        from ..pictures.codes import fact_of, word_ids
        from ..pictures.gen import draw, load_generator
        from ..pictures.reader import judge
        from ..pictures.train import load_tokenizer
        rel = str(spec.get("checkpoint") or "")
        if rel not in {g["rel"] for g in self.generators()}:
            raise PicturesError(f"not a G1 picture generator: {rel!r}")
        caption = " ".join(str(spec.get("caption") or "").lower().split())
        try:
            word_ids(caption)
        except ValueError as e:
            raise PicturesError(str(e)) from e
        n = int(spec.get("n") or 8)
        guidance = float(spec.get("guidance") if spec.get("guidance") is not None else 3.0)
        seed = int(spec.get("seed") or 0)
        if not 1 <= n <= 32 or not 0 <= guidance <= 10:
            raise PicturesError("pictures 1–32, guidance 0–10")
        cache = getattr(self, "_g1", None)
        if not cache or cache[0] != rel:
            model, pg, blob = load_generator(self.root / rel, "cpu")
            tok = Path(pg.tokenizer)
            tokenizer, _ = load_tokenizer(tok if tok.is_absolute() else self.root / tok, "cpu")
            self._g1 = cache = (rel, model, pg, blob, tokenizer)
        _, model, pg, blob, tokenizer = cache
        pics, _ = draw(model, tokenizer, [caption] * n, pg=pg, guidance=guidance, seed=seed)
        want = fact_of(caption)
        out = []
        for img in pics:
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            v = judge(img, want)
            out.append({"png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                        "ok": bool(v["all"]), "read": v["read"]})
        g = f"{guidance:g}"
        return finite({"caption": caption, "step": blob.get("step"), "pictures": out,
                       "agree": sum(p["ok"] for p in out),
                       "held_out": (want["colour"], want["shape"]) == ("purple", "triangle"),
                       "command": f'python -m aksharallm.pictures draw {rel} "{caption}" '
                                  f"--n {n} --guidance {g} --seed {seed}"})

    def start(self, spec: dict) -> dict:
        kind = spec.get("kind")
        if kind == "gen-report":
            rel = str(spec.get("checkpoint") or "")
            if rel not in {g["rel"] for g in self.generators()}:
                raise JobError(f"not a G1 picture generator: {rel!r}")
            k = int(spec.get("k") or 8)
            guidance = float(spec.get("guidance") if spec.get("guidance") is not None else 3.0)
            if not 1 <= k <= 32 or not 0 <= guidance <= 10:
                raise JobError("pictures per caption 1–32, guidance 0–10")
            return self.jobs.start(["gen-report", rel, "--k", str(k), "--guidance", f"{guidance:g}",
                                    "--device", "cpu"],
                                   f"G1 report on {rel}, {k} per caption, guidance {guidance:g}")
        if kind == "codes":
            rel = str(spec.get("checkpoint") or "")
            if rel not in {c["rel"] for c in self.checkpoints()}:
                raise JobError(f"not a picture tokenizer checkpoint: {rel!r}")
            books = int(spec.get("books") or 2)
            run = Path(rel).parent.name
            return self.jobs.start(["codes", rel, "--books", str(books),
                                    "--out", f"data/pictures/codes-{run}"],
                                   f"encode the corpus with {rel}, {books} codebooks")
        if kind == "report":
            rel = str(spec.get("checkpoint") or "")
            if rel not in {c["rel"] for c in self.checkpoints()}:
                raise JobError(f"not a picture tokenizer checkpoint: {rel!r}")
            n = int(spec.get("n") or 256)
            if not 16 <= n <= 2048:
                raise JobError("pictures must be between 16 and 2,048")
            # Always the CPU from the browser: a report is a 4.5M-parameter conv net over a few
            # hundred 64x64 pictures (seconds), and a button must never be what puts work on
            # a card someone else is using. The night queue passes --device cuda itself.
            return self.jobs.start(["report", rel, "--n", str(n), "--device", "cpu"],
                                   f"report on {rel}, {n} pictures")
        if kind == "read":
            return self.jobs.start(["read"], "the reader on the corpus (must be 100%)")
        raise JobError(f"unknown job {kind!r}")
