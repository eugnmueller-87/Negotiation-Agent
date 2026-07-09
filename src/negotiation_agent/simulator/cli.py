"""`neg-sim` — run the reference scenario matrix and print the metrics report.

    python -m negotiation_agent.simulator.cli
    neg-sim            # once installed
"""

from __future__ import annotations

import argparse
import json
import sys

# Windows consoles default to cp1252 and choke on non-ASCII. Force UTF-8 so the
# report renders identically everywhere.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from negotiation_agent.engine import EngineConfig
from negotiation_agent.simulator.metrics import BatchMetrics, run_batch
from negotiation_agent.simulator.scenarios import reference_matrix, zopa_check


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "  n/a"
    return f"{x * 100:5.1f}%" if pct else f"{x:6.3f}"


def _print_batch(title: str, b: BatchMetrics, indent: int = 0) -> None:
    pad = "  " * indent
    print(f"{pad}{title}  (n={b.n})")
    print(f"{pad}  closure     {_fmt(b.closure_rate, pct=True)}"
          f"   escalation {_fmt(b.escalation_rate, pct=True)}"
          f"   walk {_fmt(b.walk_rate, pct=True)}")
    print(f"{pad}  capture μ   {_fmt(b.capture_ratio_mean)}"
          f"   min {_fmt(b.capture_ratio_min)}"
          f"   joint μ {_fmt(b.joint_utility_mean)}")
    print(f"{pad}  rounds μ    {_fmt(b.rounds_to_close_mean)}"
          f"   median {_fmt(b.rounds_to_close_median)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the negotiation reference simulation.")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    cfg = EngineConfig(max_rounds=args.max_rounds, beta=args.beta)
    scenarios = reference_matrix()
    metrics, batch = run_batch(scenarios, cfg)

    if args.json:
        print(json.dumps({
            "batch": batch.model_dump(),
            "per_negotiation": [m.model_dump() for m in metrics],
        }, indent=2))
        return 0

    zopa = zopa_check(scenarios[0].buyer_envelope, scenarios[0].supplier_envelope)
    print("=" * 64)
    print("NEGOTIATION AGENT v0 — reference simulation")
    print(f"  ZOPA check (supplier U at buyer reservation): {zopa:.3f} "
          f"(supplier reservation {scenarios[0].supplier_envelope.reservation_utility:.2f})")
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
            f"buyerU={m.buyer_utility_final:.3f} joint={m.joint_utility:.3f}"
            if m.closed
            else "-"
        )
        print(f"  {m.scenario:28s} {m.status:16s} rounds={m.rounds_used}  {deal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
