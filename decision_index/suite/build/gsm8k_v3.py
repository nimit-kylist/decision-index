import collections
import gzip
import hashlib
import json
import random
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq

from decision_index.suite.io import dumps, read_jsonl, sha256_file

SEED = 20261003
K = 10
WINDOW = 4
METHOD = "answer-groups-v1"
RUN_PREFIX = "v0.3-gsm8k:"
MODEL = "jev-1.13.0"
TEST_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
SOURCE_PATH = "gsm8k-rows.jsonl"


def gold_of(answer):
    return Decimal(answer.split("####")[-1].strip().replace(",", ""))


def num(n):
    return format(Decimal(n).normalize(), "f")


def tiebreak(i):
    return hashlib.sha256(f"{SEED}:order:{i}".encode()).hexdigest()


def max_run(vals):
    best = run = 1
    for a, b in zip(vals, vals[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def bands(vals, k):
    out, s, n = [], 0, len(vals)
    while s < n:
        m = 1
        while s + k * m <= n and max_run(vals[s:s + k * m]) > m:
            m += 1
        if s + k * m > n:
            while True:
                s0 = out.pop()[0] if out else 0
                m = (n - s0) // k
                if max_run(vals[s0:n]) <= m:
                    out.append((s0, m))
                    return out
                if s0 == 0:
                    raise ValueError("a value occurs more than n/k times")
        out.append((s, m))
        s += k * m
    return out


def leftover(order, golds, r):
    if not r:
        return [], []
    start = random.Random(f"{SEED}:leftover").randrange(len(order))
    scan = order[start:] + order[:start][::-1]
    pick, seen = [], set()
    for i in scan:
        if golds[i] not in seen:
            pick.append(i)
            seen.add(golds[i])
        if len(pick) == r:
            break
    top = max(golds[i] for i in pick)
    bottom = min(golds[i] for i in pick)
    above = sorted({g for g in golds.values() if g > top})
    below = sorted({g for g in golds.values() if g < bottom}, reverse=True)
    return pick, (above + below)[: K - r]


def groups(golds):
    order = sorted(golds, key=lambda i: (golds[i], tiebreak(i)))
    left, extra = leftover(order, golds, len(order) % K)
    taken = set(left)
    rest = [i for i in order if i not in taken]
    vals = [golds[i] for i in rest]
    out = []
    for s, m in bands(vals, K):
        for j in range(m):
            out.append({"cases": [rest[s + t * m + j] for t in range(K)], "extra": [], "band": [s, m]})
    if left:
        out.append({"cases": sorted(left, key=lambda i: golds[i]), "extra": extra, "band": None})
    for n, g in enumerate(out):
        g["id"] = f"G{n:03d}"
        g["values"] = [golds[i] for i in g["cases"]] + g["extra"]
        assert len(g["values"]) == K and len(set(g["values"])) == K
    return out


def four_sets(group, golds, shift):
    ring = sorted(group["values"])
    n = len(ring)
    pos = {v: p for p, v in enumerate(ring)}
    return {i: [ring[(pos[golds[i]] - shift + t) % n] for t in range(WINDOW)] for i in group["cases"]}


def option_sets(golds):
    sets = {}
    gs = groups(golds)
    shifts = []
    for b in range(0, len(gs), WINDOW):
        block = list(range(WINDOW))
        random.Random(f"{SEED}:shifts:{b // WINDOW}").shuffle(block)
        shifts.extend(block[: len(gs) - b])
    for g, shift in zip(gs, shifts):
        four = four_sets(g, golds, shift)
        for i in g["cases"]:
            assert golds[i] in four[i] and set(four[i]) <= set(g["values"])
            sets[i] = {"group": g["id"], "band": g["band"], 10: g["values"], 4: four[i], "members": sorted(g["cases"])}
    return sets


def options(gold, values, count, i):
    opts = [num(v) for v in values]
    random.Random(f"{SEED}:GSM:{i}:{count}").shuffle(opts)
    return opts, opts.index(num(gold))


def payload_sha(row):
    text = dumps({"model": MODEL, "state": row["state"], "questions": row["questions"]})
    return hashlib.sha256(text.encode()).hexdigest(), max(1, len(text) // 4)


def base_rows(rows_path):
    out = []
    with gzip.open(rows_path, "rt", encoding="utf-8") if str(rows_path).endswith(".gz") else open(rows_path, encoding="utf-8") as f:
        for line in f:
            if '"catalog_id":30,' in line:
                row = json.loads(line)
                if row["_evaluation"]["catalog_id"] == 30:
                    out.append(row)
    return out


def build(test_parquet, rows_path):
    actual = sha256_file(test_parquet)
    if actual != TEST_SHA256:
        raise ValueError(f"{test_parquet}: sha256 {actual} != {TEST_SHA256}")
    test = pq.read_table(test_parquet).to_pylist()
    golds = {i: gold_of(r["answer"]) for i, r in enumerate(test)}
    old = base_rows(rows_path)
    if len(old) != 2 * len(golds):
        raise ValueError(f"expected {2 * len(golds)} GSM8K rows in {rows_path}, found {len(old)}")
    sets = option_sets(golds)
    rows = []
    for r in old:
        i = r["metadata"]["provenance"]["source_id"]
        count = len(r["questions"]["answer"]["criteria"])
        gold = golds[i]
        assert num(gold) == r["metadata"]["gold_numeric"]
        assert r["questions"]["answer"]["criteria"][r["expected"]["answer"]] == num(gold)
        ordered, correct = options(gold, sets[i][count], count, i)
        new = {
            "id": r["id"],
            "family": r["family"],
            "split": r["split"],
            "state": r["state"],
            "questions": {"answer": {**r["questions"]["answer"], "criteria": {f"option_{j}": v for j, v in enumerate(ordered)}}},
            "expected": {"answer": f"option_{correct}"},
            "metadata": {
                "group_id": r["metadata"]["group_id"],
                "provenance": {"source": r["metadata"]["provenance"]["source"], "source_sha256": TEST_SHA256, "source_id": i, "seed": SEED},
                "adaptation_scope": r["metadata"]["adaptation_scope"],
                "gold_numeric": r["metadata"]["gold_numeric"],
                "distractor_method": METHOD,
                "answer_group": {"id": sets[i]["group"], "band": sets[i]["band"], "source_ids": sets[i]["members"]},
                "replaces_run_id": r["_evaluation"]["run_id"],
            },
        }
        h, tokens = payload_sha(new)
        e = r["_evaluation"]
        new["_evaluation"] = {
            "run_id": RUN_PREFIX + e["run_id"],
            "catalog_id": 30,
            "dataset": e["dataset"],
            "group_id": e["group_id"],
            "track": e["track"],
            "source_path": SOURCE_PATH,
            "payload_sha256": h,
            "proxy_tokens": tokens,
            "benchmark_origin": e["benchmark_origin"],
            "source_run_id": e["run_id"],
            "edition": "0.3",
        }
        rows.append(new)
    return rows


def write(rows, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plain = out_dir / SOURCE_PATH
    with plain.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(dumps(r) + "\n")
    gz = out_dir / (SOURCE_PATH + ".gz")
    with plain.open("rb") as src, gzip.open(gz, "wb", compresslevel=9) as dst:
        dst.write(src.read())
    tracks = collections.Counter(r["_evaluation"]["track"] for r in rows)
    return {"out": str(gz), "requests": len(rows), "tracks": dict(sorted(tracks.items())), "answer_groups": len({r["metadata"]["answer_group"]["id"] for r in rows}), "gsm8k_sha256": sha256_file(plain)}


def main(layout, rows_path, log=print):
    test = layout.sources / "gsm8k/main/test-00000-of-00001.parquet"
    rows = build(test, rows_path)
    report = write(rows, layout.suite / "release-v3-rebuilt")
    log(json.dumps({"event": "gsm8k_v3", **report}))
    return report


def compare_payloads(rows, reference_path):
    ref = {r["_evaluation"]["run_id"]: (r["_evaluation"]["payload_sha256"], r["expected"]) for r in read_jsonl(reference_path)}
    new = {r["_evaluation"]["run_id"]: (r["_evaluation"]["payload_sha256"], r["expected"]) for r in rows}
    return {"reference_rows": len(ref), "rebuilt_rows": len(new), "identical": ref == new}
