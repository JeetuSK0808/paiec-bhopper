"""Coordinated acquisition (v7 candidate 'skipco'): skip a candidate item that ANOTHER system of the same benchmark has already revealed.

Splits are independent per subject-benchmark pair, so an item that system B has revealed is in B's acquisition pool and is never scored for
B; revealing it again for A adds nothing to B's scored predictions, while an unrevealed item is in B's evaluation pool half the time.
Otherwise the evaluator's own random rule is reproduced exactly (sha256 of [context, input] < labels_remaining / items_remaining), and
every remaining candidate is taken when the 31-label allowance needs them. The decision never looks at outcomes.
"""
import hashlib
import json


def _uniform(context, input) -> float:
    key = json.dumps([context, input], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64


def acquisition_function(input: list, *, labeled: list | None = None, context: dict | None = None) -> bool:
    try:
        left, items = int(context["labels_remaining"]), int(context["items_remaining"])
    except Exception:
        return True
    if left <= 0:
        return False
    if items <= left:
        return True
    try:
        subject, item = input
        bench = item.get("benchmark_id")
        for entry in labeled or ():
            (s2, i2), _ = entry
            if isinstance(i2, dict) and i2.get("benchmark_id") == bench and s2 != subject and i2 == item:
                return False
    except Exception:
        pass
    try:
        return bool(_uniform(context, input) < min(1.0, left / items))
    except Exception:
        return True
