"""``neg-sim`` — run the reference simulation, print metrics, or replay a transcript.

python -m negotiation_agent.simulator.cli                    # the report
neg-sim                                                      # once installed
neg-sim --json                                              # machine-readable
neg-sim --transcript ref/aggressive/oracle                  # replay one negotiation
"""

from __future__ import annotations

import argparse
import json
import sys

# Windows consoles default to cp1252 and choke on the µ / — glyphs below. Force
# UTF-8 so the report renders identically on every platform.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from negotiation_agent.baseline import compare_at_threshold
from negotiation_agent.engine import DealEngine, EngineConfig
from negotiation_agent.simulator.loop import run_negotiation
from negotiation_agent.simulator.metrics import BatchMetrics, run_batch
from negotiation_agent.simulator.scenarios import (
    Scenario,
    _true_priorities,
    reference_matrix,
    zopa_check,
)
from negotiation_agent.simulator.supplier import ParametricSupplier
from negotiation_agent.supplier_model import SupplierModel


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "  n/a"
    return f"{x * 100:5.1f}%" if pct else f"{x:6.3f}"


def _print_batch(title: str, b: BatchMetrics, indent: int = 0) -> None:
    pad = "  " * indent
    print(f"{pad}{title}  (n={b.n})")
    print(
        f"{pad}  closure     {_fmt(b.closure_rate, pct=True)}"
        f"   escalation {_fmt(b.escalation_rate, pct=True)}"
        f"   walk {_fmt(b.walk_rate, pct=True)}"
    )
    print(
        f"{pad}  capture μ   {_fmt(b.capture_ratio_mean)}"
        f"   min {_fmt(b.capture_ratio_min)}"
        f"   joint μ {_fmt(b.joint_utility_mean)}"
    )
    print(
        f"{pad}  rounds μ    {_fmt(b.rounds_to_close_mean)}"
        f"   median {_fmt(b.rounds_to_close_median)}"
    )


def _print_transcript(scenario: Scenario, cfg: EngineConfig) -> None:
    """Replay one negotiation turn by turn — the logrolling made legible.

    Watch the engine's ``threshold`` decay while ``buyerU`` holds high and the
    supplier's ``supplierU`` climbs: that gap closing on terms the buyer weights
    lightly is the win-win trade, not a price split.
    """
    engine = DealEngine(scenario.buyer_envelope, scenario.belief, cfg)
    supplier = ParametricSupplier(scenario.supplier_envelope, scenario.persona)
    result = run_negotiation(
        scenario.buyer_envelope,
        engine,
        supplier,
        supplier_envelope=scenario.supplier_envelope,
        persona_name=scenario.persona.name,
        belief_source=scenario.belief_source,
        config=cfg,
    )

    print("=" * 78)
    print(f"TRANSCRIPT  {scenario.name}   (belief={scenario.belief_source})")
    print("=" * 78)
    print(
        f"{'seq':>3}  {'actor':<8} {'rnd':>3}  {'kind':<9}"
        f"{'threshold':>10}{'buyerU':>9}{'supU':>7}   reason"
    )
    print("-" * 78)
    for tn in result.transcript.turns:
        th = f"{tn.threshold:.3f}" if tn.threshold is not None else "     -"
        bu = f"{tn.buyer_utility:.3f}" if tn.buyer_utility is not None else "    -"
        su = f"{tn.supplier_utility:.3f}" if tn.supplier_utility is not None else "    -"
        print(
            f"{tn.seq:>3}  {tn.actor:<8} {tn.round_index:>3}  {tn.kind:<9}"
            f"{th:>10}{bu:>9}{su:>7}   {tn.reason or ''}"
        )
    print("-" * 78)
    print(
        f"OUTCOME: {result.status}"
        + (f"  ({result.escalation_reason})" if result.escalation_reason else "")
    )
    if result.final_deal is not None:
        terms = "  ".join(f"{k}={v:g}" for k, v in result.final_deal.terms.items())
        buyer_u = scenario.buyer_envelope.utility(result.final_deal)
        print(f"DEAL: {terms}   → buyer utility {buyer_u:.3f}")


def _print_baseline() -> None:
    """Head-to-head: logrolling vs uniform price-splitting at matched buyer cost.

    The headline proof. At each buyer-utility threshold, both strategies deliver
    the same buyer utility — but logrolling gives the supplier a large gain by
    conceding on the terms the supplier actually values.
    """
    sc = reference_matrix()[0]
    buyer, supplier = sc.buyer_envelope, sc.supplier_envelope
    oracle = SupplierModel(appetite=_true_priorities(supplier), source="simulator")

    print("=" * 68)
    print("LOGROLLING vs PRICE-SPLIT  —  supplier utility at matched buyer cost")
    print("=" * 68)
    print(f"{'buyerU':>8} | {'logroll':>18} | {'price-split':>18} | {'supplier':>9}")
    print(f"{'target':>8} | {'buyerU  supplierU':>18} | {'buyerU  supplierU':>18} | {'gain':>9}")
    print("-" * 68)
    gains = []
    for th in [0.95, 0.85, 0.75, 0.65, 0.55]:
        c = compare_at_threshold(buyer, supplier, th, oracle)
        gains.append(c.supplier_gain)
        print(
            f"{th:>8.2f} | {c.buyer_utility_logroll:>6.3f}  {c.supplier_utility_logroll:>8.3f}"
            f"  | {c.buyer_utility_split:>6.3f}  {c.supplier_utility_split:>8.3f}"
            f"  | {c.supplier_gain:>+9.3f}"
        )
    print("-" * 68)
    print(f"mean supplier gain at matched buyer cost: {sum(gains) / len(gains):+.3f}")
    print("(same buyer utility, better counterparty deal — the win-win humans skip)")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run the negotiation reference simulation.")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument(
        "--transcript",
        metavar="SCENARIO",
        help="replay one negotiation turn by turn (e.g. ref/aggressive/oracle)",
    )
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="show logrolling vs price-split supplier gain at matched buyer cost",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = EngineConfig(max_rounds=args.max_rounds, beta=args.beta)
    scenarios = reference_matrix()

    if args.baseline:
        _print_baseline()
        return 0

    if args.transcript:
        match = next((s for s in scenarios if s.name == args.transcript), None)
        if match is None:
            names = ", ".join(s.name for s in scenarios)
            print(f"unknown scenario {args.transcript!r}. available:\n  {names}", file=sys.stderr)
            return 2
        _print_transcript(match, cfg)
        return 0

    metrics, batch = run_batch(scenarios, cfg)

    if args.json:
        print(
            json.dumps(
                {
                    "batch": batch.model_dump(),
                    "per_negotiation": [m.model_dump() for m in metrics],
                },
                indent=2,
            )
        )
        return 0

    zopa = zopa_check(scenarios[0].buyer_envelope, scenarios[0].supplier_envelope)
    print("=" * 64)
    print("NEGOTIATION AGENT v0 — reference simulation")
    print(
        f"  ZOPA check (supplier U at buyer reservation): {zopa:.3f} "
        f"(supplier reservation {scenarios[0].supplier_envelope.reservation_utility:.2f})"
    )
    print("=" * 64)
    _print_batch("OVERALL", batch)
    print("\nby persona:")
    for name, sub in batch.by_persona.items():
        _print_batch(name, sub, indent=1)
    print("\nby belief condition:")
    for name, sub in batch.by_belief.items():
        _print_batch(name, sub, indent=1)
    print("\nper negotiation:")
    for m in metrics:
        deal = (
            f"buyerU={m.buyer_utility_final:.3f} joint={m.joint_utility:.3f}" if m.closed else "-"
        )
        print(f"  {m.scenario:28s} {m.status:16s} rounds={m.rounds_used}  {deal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
