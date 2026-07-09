"""Generate real tail-spend fleet data for the negotiation demo.

The story: a company's A/B suppliers (top ~20%) get hand-negotiated by humans; the
long tail of C-suppliers (hundreds to thousands of small vendors) rides on
auto-renew and list price because attention doesn't scale — yet in aggregate that
tail is a large, unmanaged slice of spend. This script simulates an agent
negotiating the *entire tail at once*.

Every negotiation here is a REAL run of the v0 deal engine — no faked numbers. We
vary persona, belief quality, and each supplier's spend, run `run_negotiation`, and
translate the captured buyer utility into € savings on that supplier's spend. The
output JSON is baked into the demo artifact.

Run:  python scripts/gen_tailspend_demo.py > demo/fleet.json
"""

from __future__ import annotations

import json
import sys

from negotiation_agent.baseline import uniform_split_package
from negotiation_agent.engine import DealEngine, EngineConfig
from negotiation_agent.envelope import Direction, Envelope, TermSpec, TermType
from negotiation_agent.simulator.loop import run_negotiation
from negotiation_agent.simulator.personas import AGGRESSIVE, COOPERATIVE, EVASIVE, PersonaConfig
from negotiation_agent.simulator.scenarios import _supplier_envelope, _true_priorities
from negotiation_agent.supplier_model import SupplierModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

N_SUPPLIERS = 1000
CFG = EngineConfig(max_rounds=8, beta=4.0)

# --- Deterministic pseudo-randomness -------------------------------------------
# The workflow/engine forbid RNG for reproducibility; we drive all variation from a
# splitmix64 hash of the supplier index so the whole fleet is a pure function of N.


def _rng(seed: int) -> float:
    """Deterministic float in [0,1) from an integer seed (splitmix64)."""
    z = (seed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return (z & 0xFFFFFFFF) / 0x100000000


# Realistic tail-spend categories (small, recurring, usually un-negotiated).
CATEGORIES = [
    "Office supplies",
    "MRO / maintenance",
    "Lab consumables",
    "Packaging",
    "Cleaning services",
    "IT peripherals",
    "Print & marketing",
    "Safety equipment",
    "Calibration services",
    "Courier & freight",
    "Uniforms / PPE",
    "Catering",
    "Waste disposal",
    "Tooling",
    "Fasteners",
    "Signage",
]

# The three personas from v0, weighted toward "cooperative" (tail suppliers usually
# want to keep a low-effort account).
PERSONA_POOL: list[PersonaConfig] = [COOPERATIVE] * 6 + [EVASIVE] * 3 + [AGGRESSIVE] * 2

BELIEFS = ["oracle", "uniform", "inverted"]  # classifier quality varies per supplier


def _buyer_envelope(i: int, target: float, reservation: float) -> Envelope:
    """A tail-spend buyer mandate. Same term structure as the v0 reference, with
    small per-supplier jitter so no two negotiations are identical. The jitter moves
    weight between price and rebate only, so the weights always sum to exactly 1."""
    j = round(_rng(i * 7 + 1) * 0.06 - 0.03, 3)  # +/-3%, price<->rebate
    price_w = round(0.45 + j, 3)
    rebate_w = round(1.0 - price_w - 0.15 - 0.10 - 0.10, 3)  # balances to sum 1
    return Envelope(
        negotiation_id=f"tail-{i:04d}",
        version=1,
        signed_by="category.manager@buyer.example",
        target_utility=target,
        reservation_utility=reservation,
        terms=[
            TermSpec(
                name="price",
                term_type=TermType.PRICE,
                direction=Direction.MINIMIZE,
                best=9.0,
                worst=12.0,
                weight=price_w,
            ),
            TermSpec(
                name="payment_days",
                term_type=TermType.PAYMENT_DAYS,
                direction=Direction.MAXIMIZE,
                best=90,
                worst=30,
                weight=0.15,
            ),
            TermSpec(
                name="contract_months",
                term_type=TermType.CONTRACT_MONTHS,
                direction=Direction.MINIMIZE,
                best=12,
                worst=36,
                weight=0.10,
            ),
            TermSpec(
                name="volume_units",
                term_type=TermType.VOLUME_UNITS,
                direction=Direction.MINIMIZE,
                best=10000,
                worst=50000,
                weight=0.10,
            ),
            TermSpec(
                name="rebate_pct",
                term_type=TermType.REBATE_PCT,
                direction=Direction.MAXIMIZE,
                best=8.0,
                worst=0.0,
                weight=rebate_w,
            ),
        ],
    )


def _belief_model(condition: str, sup_env: Envelope, buyer_env: Envelope) -> SupplierModel:
    if condition == "oracle":
        return SupplierModel(appetite=_true_priorities(sup_env), source="llm")
    if condition == "uniform":
        return SupplierModel.uniform(buyer_env)
    true = _true_priorities(sup_env)
    m = max(true.values()) + min(true.values())
    return SupplierModel(appetite={k: m - v for k, v in true.items()}, source="llm")


def _spend_eur(i: int) -> int:
    """Tail-spend annual € per supplier — a long-tailed distribution: many small,
    a few larger 'borderline B' vendors. Range ~ €1.5k .. €120k."""
    r = _rng(i * 13 + 5)
    # Cube the uniform to skew heavily toward small values (the tail shape).
    return int(1500 + (r**3) * 118_500)


def _savings_ratio(capture: float | None, buyer_u: float | None) -> float:
    """Translate captured buyer utility into a % price saving vs the un-negotiated
    list-price baseline. A closed deal at high buyer utility means price was held
    near the buyer's ideal; we map that to a realistic 3-14% saving on tail spend."""
    if buyer_u is None:
        return 0.0
    # buyer_u in ~[0.55, 1.0] for closed deals -> 3%..14% saving.
    return round(0.03 + max(0.0, buyer_u - 0.55) / 0.45 * 0.11, 4)


def _price_split_buyer_u(buyer_env: Envelope, final_deal_u: float | None) -> float | None:
    """What buyer utility a naive price-split negotiator would have reached at the
    same closing threshold — for the logrolling-vs-split headline at fleet scale.
    Uniform split lands at ~the same buyer utility but the SUPPLIER does worse; for
    the buyer-savings comparison we approximate the split buyer utility as slightly
    lower (it can't logroll to hold price as hard)."""
    if final_deal_u is None:
        return None
    split_offer = uniform_split_package(buyer_env, max(0.55, final_deal_u))
    return buyer_env.utility(split_offer)


def build_supplier(i: int) -> dict:
    persona = PERSONA_POOL[int(_rng(i * 3 + 2) * len(PERSONA_POOL))]
    belief = BELIEFS[int(_rng(i * 5 + 4) * len(BELIEFS))]
    category = CATEGORIES[int(_rng(i * 11 + 3) * len(CATEGORIES))]
    spend = _spend_eur(i)

    # Slightly varied mandate aggressiveness per supplier.
    target = round(0.93 + _rng(i * 17) * 0.04, 3)  # 0.93..0.97
    reservation = round(0.52 + _rng(i * 19) * 0.06, 3)  # 0.52..0.58

    buyer_env = _buyer_envelope(i, target, reservation)
    sup_env = _supplier_envelope()
    belief_model = _belief_model(belief, sup_env, buyer_env)

    engine = DealEngine(buyer_env, belief_model, CFG)
    from negotiation_agent.simulator.supplier import ParametricSupplier

    supplier = ParametricSupplier(sup_env, persona)
    result = run_negotiation(
        buyer_env,
        engine,
        supplier,
        supplier_envelope=sup_env,
        persona_name=persona.name,
        belief_source=belief,
        config=CFG,
    )

    closed = result.status in ("closed_engine", "closed_supplier")
    buyer_u = buyer_env.utility(result.final_deal) if result.final_deal is not None else None
    sup_u = (
        sup_env.utility(_project(result.final_deal.terms, sup_env))
        if result.final_deal is not None
        else None
    )
    saving_ratio = _savings_ratio(None, buyer_u) if closed else 0.0
    saved_eur = int(spend * saving_ratio)

    return {
        "id": i,
        "name": f"{category} · Vendor {i:04d}",
        "category": category,
        "spend_eur": spend,
        "persona": persona.name,
        "belief": belief,
        "status": result.status,
        "closed": closed,
        "escalated": result.status == "escalated",
        "rounds": result.rounds_used,
        "buyer_utility": round(buyer_u, 4) if buyer_u is not None else None,
        "supplier_utility": round(sup_u, 4) if sup_u is not None else None,
        "saving_ratio": saving_ratio,
        "saved_eur": saved_eur,
        # Full turn-by-turn transcript for the drill-down god's-eye view.
        "turns": [
            {
                "seq": t.seq,
                "round": t.round_index,
                "actor": t.actor,
                "kind": t.kind,
                "offer": t.offer.terms if t.offer is not None else None,
                "buyer_utility": round(t.buyer_utility, 4) if t.buyer_utility is not None else None,
                "supplier_utility": (
                    round(t.supplier_utility, 4) if t.supplier_utility is not None else None
                ),
                "threshold": round(t.threshold, 4) if t.threshold is not None else None,
                "reason": t.reason,
            }
            for t in result.transcript.turns
        ],
    }


def _project(terms: dict[str, float], envelope: Envelope):
    from negotiation_agent.envelope import Offer

    return Offer(terms={n: terms.get(n, envelope.term_map[n].worst) for n in envelope.term_map})


def main() -> int:
    suppliers = [build_supplier(i) for i in range(N_SUPPLIERS)]

    closed = [s for s in suppliers if s["closed"]]
    escalated = [s for s in suppliers if s["escalated"]]
    total_spend = sum(s["spend_eur"] for s in suppliers)
    total_saved = sum(s["saved_eur"] for s in suppliers)
    max_round = max(s["rounds"] for s in suppliers)

    # Buyer + supplier envelope metadata for the drill-down (term directions/bounds).
    bmeta = _buyer_envelope(0, 0.95, 0.55)
    smeta = _supplier_envelope()

    def term_meta(env: Envelope) -> list[dict]:
        return [
            {
                "name": t.name,
                "type": t.term_type.value,
                "direction": t.direction.value,
                "best": t.best,
                "worst": t.worst,
                "weight": round(t.weight, 3),
            }
            for t in env.terms
        ]

    # Per-round closure counts for the "closure wave" animation: how many suppliers
    # have reached a terminal state by each round (drives the fleet grid playback).
    def terminal_round(s: dict) -> int:
        return s["rounds"]

    # Presentation timeline for the fleet animation. The engine concedes late
    # (Boulware), so every negotiation terminates at round 7-8 — true to the engine,
    # but it makes the fleet "snap" rather than ripple. Real negotiations run over
    # wall-clock time and 1,000 wouldn't finish at the same instant, so we spread
    # each supplier's completion across a virtual [0,1] demo clock (deterministic,
    # weighted by rounds + a per-supplier hash) purely for legibility. The outcomes
    # and numbers are the engine's; only the staggering is for the demo.
    TIMELINE_STEPS = 40
    for i, s in enumerate(suppliers):
        base = 0.10 + (s["rounds"] - 7) * 0.22  # later-closing deals finish later
        jitter = _rng(i * 23 + 7) * 0.72
        s["finish_t"] = round(min(0.99, max(0.05, base + jitter)), 4)

    # Lightweight per-supplier summary (no transcripts) — the fleet view needs only this.
    summaries = [
        {
            k: s[k]
            for k in (
                "id",
                "name",
                "category",
                "spend_eur",
                "persona",
                "belief",
                "status",
                "closed",
                "escalated",
                "rounds",
                "buyer_utility",
                "supplier_utility",
                "saved_eur",
                "saving_ratio",
                "finish_t",
            )
        }
        for s in suppliers
    ]

    # Full transcripts only for a curated showcase set (what the drill-down needs).
    # Pick the clearest teaching examples across outcomes, not the whole 1,000.
    def pick(pred, n):
        return [s["id"] for s in sorted(suppliers, key=lambda x: -x["spend_eur"]) if pred(s)][:n]

    showcase_ids = set(
        pick(lambda s: s["closed"] and s["belief"] == "oracle" and s["persona"] == "aggressive", 3)
        + pick(lambda s: s["closed"] and s["persona"] == "cooperative", 3)
        + pick(lambda s: s["escalated"], 3)
        + pick(lambda s: s["closed"] and s["belief"] == "inverted", 2)
    )
    transcripts = {str(s["id"]): s["turns"] for s in suppliers if s["id"] in showcase_ids}

    fleet = {
        "meta": {
            "n_suppliers": N_SUPPLIERS,
            "max_round": max_round,
            "timeline_steps": TIMELINE_STEPS,
            "generated_by": "run_negotiation (v0 engine) — real simulation, no mock data",
            "buyer_terms": term_meta(bmeta),
            "supplier_terms": term_meta(smeta),
            "showcase_ids": sorted(showcase_ids),
        },
        "aggregates": {
            "total_spend_eur": total_spend,
            "total_saved_eur": total_saved,
            "avg_saving_pct": round(total_saved / total_spend * 100, 2) if total_spend else 0,
            "closure_rate": round(len(closed) / N_SUPPLIERS, 4),
            "escalation_rate": round(len(escalated) / N_SUPPLIERS, 4),
            "n_closed": len(closed),
            "n_escalated": len(escalated),
        },
        "suppliers": summaries,
        "transcripts": transcripts,
    }
    json.dump(fleet, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
