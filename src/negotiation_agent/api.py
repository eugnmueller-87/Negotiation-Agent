"""Optional HTTP surface for the v1 pre-flight — a thin FastAPI adapter.

This is the *only* place a web framework enters the package, and it is opt-in:
FastAPI is a `[web]` extra, not a runtime dependency. Importing this module
without FastAPI installed raises a clear message rather than a bare ImportError.

The single endpoint, ``POST /prepare``, takes a contract and returns the extracted
opening position plus (optionally) a supplier due-diligence brief. It constructs
the :class:`~negotiation_agent.research.HadesClient` server-side so the ``HADES_API_KEY``
credential lives only on the server — never in the browser, per the confidentiality
line this project holds.

Run locally (after ``pip install -e ".[web]"``)::

    uvicorn negotiation_agent.api:app --reload
"""

from __future__ import annotations

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The HTTP API needs the 'web' extra. Install it with: pip install -e '.[web]'"
    ) from e

from negotiation_agent.prepare import PreparedNegotiation, prepare_negotiation
from negotiation_agent.research import HadesClient

app = FastAPI(title="Negotiation Agent — pre-flight API", version="1.0")


class PrepareRequest(BaseModel):
    """A contract to read and an opt-in flag for supplier research."""

    contract_text: str
    research: bool = True


@app.post("/prepare", response_model=PreparedNegotiation)
def prepare(req: PrepareRequest) -> PreparedNegotiation:
    """Extract the opening position from a contract and brief the supplier.

    The ``HadesClient`` reads its credential from the environment; if none is set,
    research degrades to a note and the extraction is still returned. The engine is
    never touched here — this only prepares the buyer's starting state.
    """
    researcher = HadesClient()
    return prepare_negotiation(req.contract_text, researcher=researcher, research=req.research)
