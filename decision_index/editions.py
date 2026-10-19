import hashlib
import json
from importlib import resources

DEFAULT = "0.3"

EDITIONS = {
    "0.1": dict(
        id="0.1",
        name="release-v1",
        label="Decision Index 0.1",
        panel_id="core25-observed-protocol-v1",
        headline="balanced_raw",
        suite_dir="suite",
        dataset="multimodalart/decision-index-suite",
        rows_gz_sha256="750d353a3a83af615c67cfe9752e005bf09e6c28d9c4ba28d3a9f57ba8536cfd",
        rows_sha256="288d37207a9581187bdf83eada1983aa63de6fc50b0108e2badb229547a57f99",
        added_sha256=None,
        exclusions_sha256="331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81",
        acos_subset_sha256=None,
        rows_file_requests=132422,
        requests=132422,
        added_requests=0,
        excluded=442,
        scoreable=131980,
        benchmarks=37,
    ),
    "0.2": dict(
        id="0.2",
        name="release-v2",
        label="Decision Index 0.2",
        panel_id="decision-index-0.2",
        headline="balanced_skill",
        suite_dir="suite-0.2",
        dataset="multimodalart/decision-index-suite-0.2",
        rows_gz_sha256="25aac5e890a54a3172c7a0c184b4cc8b9a43f10b6ee89bbad8da923be423c656",
        rows_sha256="b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5",
        added_sha256="7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89",
        exclusions_sha256="331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81",
        acos_subset_sha256="3b20eea1613ae3e1644339e6f89a4287b307a5750a86239bb8bca6f8238b8e45",
        retrieval_subsets_sha256="9228a9492ea40c499d8bb024417f566d0c0ced62c71ed4c1bf908e87ed50d8df",
        rows_file_requests=124971,
        requests=121057,
        added_requests=30419,
        excluded=442,
        scoreable=120615,
        benchmarks=44,
    ),
    "0.2.1": dict(
        id="0.2.1",
        name="release-v2.1",
        label="Decision Index 0.2.1",
        panel_id="decision-index-0.2.1",
        headline="balanced_skill",
        suite_dir="suite-0.2",
        dataset="multimodalart/decision-index-suite-0.2",
        rows_gz_sha256="25aac5e890a54a3172c7a0c184b4cc8b9a43f10b6ee89bbad8da923be423c656",
        rows_sha256="b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5",
        added_sha256="7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89",
        exclusions_sha256="331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81",
        acos_subset_sha256="3b20eea1613ae3e1644339e6f89a4287b307a5750a86239bb8bca6f8238b8e45",
        retrieval_subsets_sha256="9228a9492ea40c499d8bb024417f566d0c0ced62c71ed4c1bf908e87ed50d8df",
        toolret_subset_sha256="c301d5503955c14d99485e284d434bb1533f51229c34da1edc19dd8388116f01",
        bright_subset_sha256="ed3a05220e09be2b96c02514345c61894f7a0ffc73e6884a100392d23692cfb8",
        home_appliances_subset_sha256="d2d0df922a47a9757bef17f382a4ed4c5c3bfff953f941c29404e93a490bce2c",
        rows_file_requests=124971,
        requests=120340,
        added_requests=30419,
        excluded=442,
        scoreable=119898,
        benchmarks=44,
    ),
    "0.3": dict(
        id="0.3",
        name="release-v3",
        label="Decision Index 0.3",
        panel_id="decision-index-0.3",
        headline="balanced_skill",
        suite_dir="suite-0.3",
        dataset="multimodalart/decision-index-suite-0.3",
        rows_gz_sha256="25aac5e890a54a3172c7a0c184b4cc8b9a43f10b6ee89bbad8da923be423c656",
        rows_sha256="b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5",
        added_sha256="7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89",
        gsm8k_sha256="759334894858a37d6bd7bd07298eac8e55da236e31dad327ccd06a9979f4738a",
        exclusions_sha256="331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81",
        acos_subset_sha256="3b20eea1613ae3e1644339e6f89a4287b307a5750a86239bb8bca6f8238b8e45",
        retrieval_subsets_sha256="9228a9492ea40c499d8bb024417f566d0c0ced62c71ed4c1bf908e87ed50d8df",
        toolret_subset_sha256="c301d5503955c14d99485e284d434bb1533f51229c34da1edc19dd8388116f01",
        bright_subset_sha256="ed3a05220e09be2b96c02514345c61894f7a0ffc73e6884a100392d23692cfb8",
        home_appliances_subset_sha256="d2d0df922a47a9757bef17f382a4ed4c5c3bfff953f941c29404e93a490bce2c",
        retired=(48,),
        replaced={30: "v0.3-gsm8k:"},
        rows_file_requests=124971,
        requests=110201,
        added_requests=30419,
        excluded=442,
        scoreable=109759,
        benchmarks=43,
    ),
}

SUBSET_FILES = {
    "0.2": {"acos_subset_sha256": "release-v2/acos-subset.json", "retrieval_subsets_sha256": "release-v2/retrieval-subsets.json"},
    "0.2.1": {
        "acos_subset_sha256": "release-v2/acos-subset.json",
        "retrieval_subsets_sha256": "release-v2/retrieval-subsets.json",
        "toolret_subset_sha256": "release-v2.1/toolret-subset.json",
        "bright_subset_sha256": "release-v2.1/bright-subset.json",
        "home_appliances_subset_sha256": "release-v2.1/home-appliances-subset.json",
    },
}
SUBSET_FILES["0.3"] = SUBSET_FILES["0.2.1"]
SCORING_SUBSETS = {
    "0.2": ("release-v2/acos-subset.json",),
    "0.2.1": ("release-v2/acos-subset.json", "release-v2.1/toolret-subset.json", "release-v2.1/bright-subset.json", "release-v2.1/home-appliances-subset.json"),
}
SCORING_SUBSETS["0.3"] = SCORING_SUBSETS["0.2.1"]

ROWS_FILE = "selected-rows.jsonl.gz"
ADDED_FILE = "added-rows.jsonl.gz"
GSM8K_FILE = "gsm8k-rows.jsonl.gz"
EXCLUSIONS_FILE = "excluded-questions.json"
MANIFEST_FILE = "manifest.json"


def get(edition=None):
    key = str(edition or DEFAULT)
    key = {"release-v1": "0.1", "release-v2": "0.2", "release-v2.1": "0.2.1", "release-v3": "0.3", "v0.1": "0.1", "v0.2": "0.2", "v0.2.1": "0.2.1", "v0.3": "0.3"}.get(key, key)
    if key not in EDITIONS:
        raise ValueError(f"unknown edition {edition!r}; choose from {sorted(EDITIONS)}")
    return EDITIONS[key]


def compatible(edition, found):
    a, b = get(edition), get(found)
    return (a["rows_sha256"], a["added_sha256"], a.get("gsm8k_sha256")) == (b["rows_sha256"], b["added_sha256"], b.get("gsm8k_sha256"))


def v2_family(edition):
    return get(edition)["added_sha256"] is not None


def files(edition):
    e = get(edition)
    return (ROWS_FILE, EXCLUSIONS_FILE, MANIFEST_FILE) + ((ADDED_FILE,) if e["added_sha256"] else ()) + ((GSM8K_FILE,) if e.get("gsm8k_sha256") else ())


def data(name):
    return json.loads(resources.files("decision_index").joinpath("data", name).read_text())


def acos_subset():
    return data("release-v2/acos-subset.json")


def retrieval_subsets():
    return data("release-v2/retrieval-subsets.json")


def subset_sha256(edition):
    return {key: hashlib.sha256(resources.files("decision_index").joinpath("data", name).read_bytes()).hexdigest() for key, name in SUBSET_FILES.get(get(edition)["id"], {}).items()}


def scoring_subsets(edition):
    return {s["catalog_id"]: frozenset(s["run_ids"]) for s in map(data, SCORING_SUBSETS.get(get(edition)["id"], ()))}


def scoring_subset(edition):
    if get(edition)["id"] != "0.2":
        return None
    subset = acos_subset()
    return subset["catalog_id"], frozenset(subset["run_ids"])


def in_edition(edition):
    ed = get(edition)
    keep = scoring_subsets(edition)
    retired = frozenset(ed.get("retired", ()))
    replaced = dict(ed.get("replaced", {}))

    def ok(e):
        n = e["catalog_id"]
        if n in retired:
            return False
        if n in replaced and not e["run_id"].startswith(replaced[n]):
            return False
        return n not in keep or e["run_id"] in keep[n]

    return ok
