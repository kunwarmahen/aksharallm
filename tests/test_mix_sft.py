"""`data.mix_sft`: combine SFT sets by fraction, and drop HumanEval overlap.

The code SFT for the Python specialist is judged on HumanEval, so a training block that
contains a HumanEval prompt or solution would make that number meaningless -- and nothing in
the loss would say so. The first test is the positive control: a planted HumanEval prompt
must be found. A decontaminator that finds nothing because it is broken looks exactly like
a clean dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aksharallm.data.mix_sft import mix, parse_part

ROOT = Path(__file__).resolve().parents[1]
TOK = ROOT / "data" / "tinystories" / "tokenizer.json"
pytestmark = pytest.mark.skipif(not TOK.exists(), reason="needs a tokenizer on disk")

SEQ = 256
HE = [{
    "task_id": "HumanEval/0",
    "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], '
              'threshold: float) -> bool:\n    """ Check if in given list of numbers, are any '
              'two numbers closer to each other than\n    given threshold.\n    """\n',
    "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in "
                          "enumerate(numbers):\n            if idx != idx2:\n                "
                          "distance = abs(elem - elem2)\n                if distance < "
                          "threshold:\n                    return True\n\n    return False\n",
    "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) "
            "== True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n",
}]


@pytest.fixture(scope="module")
def tok():
    from aksharallm.tokenizer.tokenizer import Tokenizer
    return Tokenizer(str(TOK))


def block(tok, text: str) -> np.ndarray:
    ids = tok.encode(text)
    ids = (ids * (SEQ // max(1, len(ids)) + 1))[:SEQ]
    return np.asarray(ids, dtype=np.uint16)


def write_part(d: Path, blocks: list[np.ndarray], val: list[np.ndarray] | None = None):
    d.mkdir(parents=True)
    for split, bs in (("train", blocks), ("val", val or blocks[:1])):
        t = np.stack(bs)
        np.save(d / f"{split}_tokens.npy", t)
        np.save(d / f"{split}_mask.npy", (t % 2).astype(np.uint8))   # a mask we can check


CLEAN = ["def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n",
         "The sun is a star. Plants turn its light into sugar, and animals eat the plants.",
         "for i in range(10):\n    if i % 3 == 0:\n        print('fizz', i)\n"]


def test_a_block_holding_a_humaneval_prompt_is_dropped_and_named(tmp_path, tok):
    planted = block(tok, "Here is a task.\n" + HE[0]["prompt"] + "Please solve it.")
    write_part(tmp_path / "a", [block(tok, c) for c in CLEAN] + [planted])
    man = mix([(tmp_path / "a", 1.0)], tmp_path / "out", tok, HE)
    train = [p for p in man["parts"] if p["split"] == "train"][0]
    assert train["blocks_dropped"] == 1
    assert train["humaneval_tasks_hit"] == ["HumanEval/0"]
    assert train["blocks_strong"] == 1, "a whole copied prompt is a strong hit, not an idiom"
    kept = np.load(tmp_path / "out" / "train_tokens.npy")
    assert len(kept) == 3
    assert not any(np.array_equal(k, planted) for k in kept)


def test_a_solution_or_its_tests_alone_also_count(tmp_path, tok):
    """The answer without its prompt is what a scraped "solutions" file looks like."""
    write_part(tmp_path / "a", [block(tok, HE[0]["canonical_solution"]),
                                block(tok, HE[0]["test"]), block(tok, CLEAN[0])])
    man = mix([(tmp_path / "a", 1.0)], tmp_path / "out", tok, HE)
    assert man["parts"][0]["blocks_dropped"] == 2


def test_ordinary_code_is_not_flagged(tmp_path, tok):
    """`for idx, elem in enumerate(...)` alone must not be HumanEval -- 13 tokens of common
    Python would make every code block dirty and the filter would delete the dataset."""
    common = block(tok, "for idx, elem in enumerate(items):\n    print(idx, elem)\n")
    write_part(tmp_path / "a", [common] + [block(tok, c) for c in CLEAN])
    man = mix([(tmp_path / "a", 1.0)], tmp_path / "out", tok, HE)
    assert man["parts"][0]["blocks_dropped"] == 0


def test_fractions_take_that_share_and_masks_stay_aligned(tmp_path, tok):
    a = [block(tok, f"x = {i}\nprint(x * {i})\n") for i in range(40)]
    b = [block(tok, f"Number {i} is a number between {i - 1} and {i + 1}.") for i in range(20)]
    write_part(tmp_path / "a", a)
    write_part(tmp_path / "b", b)
    man = mix([(tmp_path / "a", 1.0), (tmp_path / "b", 0.5)], tmp_path / "out", tok, HE)
    t = np.load(tmp_path / "out" / "train_tokens.npy")
    m = np.load(tmp_path / "out" / "train_mask.npy")
    assert len(t) == 40 + 10 == man["splits"]["train"]["blocks"]
    assert np.array_equal(m, (t % 2).astype(np.uint8)), "a mask was shuffled away from its block"
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["n"] == 13


def test_parse_part():
    assert parse_part("data/sft:0.4") == (Path("data/sft"), 0.4)
    assert parse_part("data/sft") == (Path("data/sft"), 1.0)
    with pytest.raises(ValueError):
        parse_part("data/sft:1.5")


def test_a_counting_list_is_not_humaneval(tmp_path, tok):
    """Measured on the first real mix: 13 tokens of this tokenizer can be `1, 2, 3, 4, 5`,
    and HumanEval's tests are full of such lists -- 19% of a Python instruction set and 5% of
    plain chat data "matched" before n-grams without words stopped counting."""
    he = [{**HE[0], "test": "def check(candidate):\n    assert candidate([1, 2, 3, 4, 5, 6, 7, "
                            "8, 9, 10]) == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\n"}]
    lists = block(tok, "nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\ngrid = [[1, 2, 3], [4, 5, 6]]\n")
    write_part(tmp_path / "a", [lists, block(tok, CLEAN[0])])
    man = mix([(tmp_path / "a", 1.0)], tmp_path / "out", tok, he)
    assert man["parts"][0]["blocks_dropped"] == 0
