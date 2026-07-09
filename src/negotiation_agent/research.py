"""Supplier research — a due-diligence brief for the buyer, via the Hades agent.

Before negotiating, the buyer's team wants to know who they're dealing with:
sanctions exposure, registry status, LkSG/CSDDD compliance signals, ESG/labour
risk, recent news. That research is done by **Hades**, a separate deployed
service (FastAPI on Railway) that runs six parallel research pipelines and
returns a structured risk report.

This module is the *client*. It calls Hades server-side (the API key never
leaves the server — it must never reach a browser), maps the response into a
typed :class:`SupplierBrief`, and hands that to the buyer as **context**.

Design boundary, on purpose: the brief **informs the human**, it does not feed
the deal engine. The engine's decisions stay a pure function of the signed
envelope and the offers on the table — auditable, replayable, uninfluenced by
inferred external data. Research is advisory; the engine is authority.

Nice symmetry worth noting: Hades itself follows the same "LLM advises, code
decides" rule this project is built on — its report LLM writes the prose, but the
risk *score* and *recommendation* are computed deterministically and overwrite
the model's echo, so a hallucinated number can't reach the user.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_HADES_URL = "https://hades-production-b86a.up.railway.app"
_TIMEOUT_SECONDS = 150  # Hades runs ~6 pipelines; a full investigation can take ~2 min.


class ResearchUnavailable(Exception):
    """Hades could not be reached, was not configured, or returned an error.

    Never fatal to a negotiation — the caller degrades to "no brief available"
    and the human proceeds without it. The message is safe to show a buyer; it
    never contains the API key or a raw upstream error body.
    """


class SupplierBrief(BaseModel):
    """A buyer-facing due-diligence summary. Advisory context, not engine input."""

    model_config = {"frozen": True}

    company: str
    risk_score: float | None = Field(
        default=None, description="overall risk 1-10 (higher = riskier)"
    )
    risk_level: str | None = None  # Low | Medium | High | Critical
    recommendation: str | None = None  # Approve | Conditional Approval | Block
    executive_summary: str = ""
    sanctioned: bool | None = None
    sanctions_note: str = ""
    registry_status: str | None = None  # active | dissolved/insolvent | unknown
    legal_name: str | None = None
    lksg_signal: str | None = None  # no_findings | needs_monitoring | red_flag
    esg_rating: str | None = None  # positive | neutral | medium_risk | high_risk
    news_sentiment: str | None = None
    next_steps: list[str] = Field(default_factory=list)
    source: Literal["hades", "sample"] = "hades"

    @property
    def is_blocking(self) -> bool:
        """True if the recommendation says do not proceed without escalation."""
        return (self.recommendation or "").lower().startswith("block")

    def headline(self) -> str:
        """One-line summary for a UI badge or a minutes line."""
        score = f"{self.risk_score:.1f}/10" if self.risk_score is not None else "—"
        level = self.risk_level or "risk unknown"
        rec = self.recommendation or "no recommendation"
        return f"{self.company}: {level} ({score}) · {rec}"


def brief_from_hades_response(payload: dict[str, object]) -> SupplierBrief:
    """Map a raw Hades ``/investigate`` response into a :class:`SupplierBrief`.

    Tolerant of missing keys (Hades fills nulls, never omits, but we don't rely
    on that). Reads the *deterministic* verdict fields, which Hades grounds on
    code rather than the model's prose.
    """

    def sub(d: dict[str, object], key: str) -> dict[str, object]:
        v = d.get(key)
        return v if isinstance(v, dict) else {}

    report = sub(payload, "report")
    company_v = payload.get("company") or report.get("company")
    company = company_v if isinstance(company_v, str) and company_v else "unknown supplier"

    overview = sub(report, "company_overview")
    sanctions = sub(report, "sanctions_status")
    lksg = sub(report, "lksg_csddd_assessment")
    esg = sub(report, "esg_labour")
    news = sub(report, "news_sentiment")

    return SupplierBrief(
        company=company,
        risk_score=_as_float(report.get("overall_risk_score")),
        risk_level=_str(report.get("risk_level")),
        recommendation=_str(report.get("recommendation")),
        executive_summary=_str(report.get("executive_summary")) or "",
        sanctioned=_bool(sanctions.get("is_sanctioned")),
        sanctions_note=_str(sanctions.get("summary")) or "",
        registry_status=_str(overview.get("company_status")),
        legal_name=_str(overview.get("legal_name")),
        lksg_signal=_str(lksg.get("compliance_signal")),
        esg_rating=_str(esg.get("esg_rating")),
        news_sentiment=_str(news.get("sentiment")),
        next_steps=_str_list(report.get("required_next_steps")),
        source="hades",
    )


def _as_float(v: object) -> float | None:
    try:
        return float(v) if isinstance(v, (int, float, str)) else None
    except (TypeError, ValueError):
        return None


def _str(v: object) -> str | None:
    return v if isinstance(v, str) else None


def _bool(v: object) -> bool | None:
    return v if isinstance(v, bool) else None


def _str_list(v: object) -> list[str]:
    return [x for x in v if isinstance(x, str)] if isinstance(v, list) else []


class HadesClient:
    """Server-side client for the Hades supplier-due-diligence API.

    The API key is read from the environment (``HADES_API_KEY``) and sent in the
    ``X-API-Key`` header. It is never logged and never returned in an error. This
    client must run server-side only — putting the key in client-side code would
    expose a paid, rate-capped credential to anyone who opens the page.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = _TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = (base_url or os.environ.get("HADES_URL", DEFAULT_HADES_URL)).rstrip("/")
        self._api_key = api_key or os.environ.get("HADES_API_KEY", "")
        self.timeout = timeout

    def investigate(self, company: str, category: str = "", country: str = "DE") -> SupplierBrief:
        """Run a due-diligence investigation and return a :class:`SupplierBrief`.

        Raises :class:`ResearchUnavailable` on any failure (missing key,
        unreachable service, rate limit, timeout, malformed response) — always
        with a message safe to show a buyer.
        """
        if not self._api_key:
            raise ResearchUnavailable(
                "Supplier research is not configured (no HADES_API_KEY set on the server)."
            )
        body = json.dumps(
            {"company": company, "category": category, "country": country, "mode": "full"}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/investigate",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": self._api_key,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (trusted host)
                raw = resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise ResearchUnavailable(
                    "Supplier research is temporarily rate-limited — try again shortly."
                ) from None
            if e.code in (401, 403):
                raise ResearchUnavailable(
                    "Supplier research rejected the request (authentication)."
                ) from None
            # Never surface the upstream error body — it can leak internals.
            raise ResearchUnavailable(f"Supplier research failed (status {e.code}).") from None
        except (urllib.error.URLError, TimeoutError):
            raise ResearchUnavailable(
                "Supplier research service is unreachable right now."
            ) from None

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise ResearchUnavailable(
                "Supplier research returned an unreadable response."
            ) from None
        return brief_from_hades_response(payload)


def sample_brief(company: str = "Nordwerk Verpackung GmbH") -> SupplierBrief:
    """A realistic, clearly-labelled sample brief for demos and tests — no live call.

    Mirrors the shape a real Hades investigation returns, marked ``source="sample"``
    so it can never be mistaken for a live compliance result.
    """
    return SupplierBrief(
        company=company,
        risk_score=3.4,
        risk_level="Medium",
        recommendation="Conditional Approval",
        executive_summary=(
            f"{company} presents a moderate overall risk. No sanctions matches were found and "
            "the company is an active registered entity, but recent news carries low-severity "
            "negative signals and LkSG monitoring is advised for the category."
        ),
        sanctioned=False,
        sanctions_note="No matches on OFAC SDN or UN Consolidated List.",
        registry_status="active",
        legal_name=company,
        lksg_signal="needs_monitoring",
        esg_rating="neutral",
        news_sentiment="negative_low",
        next_steps=[
            "Request the supplier's current LkSG risk-management declaration",
            "Set a Hermes monitoring alert on the supplier ahead of contract signing",
        ],
        source="sample",
    )
