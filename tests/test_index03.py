import json
from pathlib import Path

import pytest

from decision_index import editions
from decision_index.scoring import index02 as X

BOARD = json.loads((Path(__file__).parent / "fixtures/board-0.3.json").read_text())
GOLD = [1, 3, 4, 5, 12, 25, 28, 29, 36, 45, 57, 58]


def spec():
    return X.spec("0.3")


def test_panel():
    s = spec()
    areas = {a["id"]: a["benchmarks"] for a in s["areas"]}
    ids = [n for a in s["areas"] for n in a["benchmarks"]]
    assert len(ids) == len(set(ids)) == 37
    assert list(areas) == ["knowledge", "language", "retrieval", "tools", "arts"]
    assert 28 in areas["knowledge"] and 28 not in areas["language"]
    assert 48 not in ids and not s["loss_rules"]
    assert sorted(int(n) for n in s["gold"]) == GOLD


def test_area_weights_fixed():
    w = X.area_weights(spec())
    assert {k: round(v, 6) for k, v in w.items()} == {"knowledge": 0.258494, "language": 0.258494, "retrieval": 0.200229, "tools": 0.182783, "arts": 0.1}


@pytest.mark.parametrize("engine", sorted(BOARD["entrants"]))
def test_reproduces_public_index(engine):
    entrant = BOARD["entrants"][engine]
    values = {int(n): dict(b) for n, b in entrant["benchmarks"].items()}
    scores, _ = X.aggregate(values, spec())
    assert scores["balanced_skill"] == pytest.approx(entrant["public_index"], abs=0.01)


def test_edition_reads_rebuilt_gsm8k_and_skips_forecastbench():
    keep = editions.in_edition("0.3")
    assert not keep({"catalog_id": 30, "run_id": "GSM8K-4:test:0"})
    assert keep({"catalog_id": 30, "run_id": "v0.3-gsm8k:GSM8K-4:test:0"})
    assert not keep({"catalog_id": 48, "run_id": "anything"})
    assert keep({"catalog_id": 25, "run_id": "anything"})
    old = editions.in_edition("0.2.1")
    assert old({"catalog_id": 30, "run_id": "GSM8K-4:test:0"}) and old({"catalog_id": 48, "run_id": "anything"})


def test_edition_files_and_compatibility():
    e = editions.get("0.3")
    assert editions.DEFAULT == "0.3" and editions.get("release-v3") is e
    assert editions.GSM8K_FILE in editions.files("0.3") and editions.GSM8K_FILE not in editions.files("0.2.1")
    assert not editions.compatible("0.3", "0.2.1") and editions.compatible("0.2.1", "0.2")
    assert editions.v2_family("0.3") and not editions.v2_family("0.1")
    assert e["requests"] + e["added_requests"] - e["excluded"] == 140178
