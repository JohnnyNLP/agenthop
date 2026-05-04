"""Shared pricing table and cost-reconstruction formula.

Used by:
  - testbed/evaluate.py   (per-run cost in summary.json)
  - cost_proof/compute_estimates.py (budget projection)
  - cost_proof/reconstruct_cost.py  (invoice reconciliation)

Pricing units: USD per 1M tokens. Caching multipliers relative to input rate.
Rates are best-effort and should be re-verified against provider dashboards
before final submission to the funding POC.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    provider: str
    input_per_m: float            # $/Mtok regular input
    output_per_m: float           # $/Mtok output
    cache_read_mult: float = 1.0  # multiplier on input_per_m for cached reads
    cache_write_mult: float = 1.0 # multiplier on input_per_m for 5-min cache writes


# Prefix-match registry. First matching prefix wins.
_REGISTRY: list[tuple[tuple[str, ...], ModelPrice]] = [
    # Anthropic
    (("claude-opus-4-6",),          ModelPrice("Anthropic", 5.00, 25.00, 0.10, 1.25)),
    (("claude-opus-4-5",),          ModelPrice("Anthropic", 15.00, 75.00, 0.10, 1.25)),
    (("claude-opus-4-1", "claude-opus-4"), ModelPrice("Anthropic", 15.00, 75.00, 0.10, 1.25)),
    (("claude-sonnet-4-6",),        ModelPrice("Anthropic", 3.00, 15.00, 0.10, 1.25)),
    (("claude-sonnet-4-5", "claude-sonnet-4"), ModelPrice("Anthropic", 3.00, 15.00, 0.10, 1.25)),
    (("claude-haiku-4-5", "claude-haiku-4"), ModelPrice("Anthropic", 1.00, 5.00, 1.00, 1.00)),  # no cache
    # OpenAI
    (("gpt-5.4",),                  ModelPrice("OpenAI", 1.25, 10.00, 0.50, 1.00)),  # auto-cache
    (("gpt-5.3-codex", "gpt-5.3"),  ModelPrice("OpenAI", 1.25, 10.00, 0.50, 1.00)),
    (("gpt-5",),                    ModelPrice("OpenAI", 1.25, 10.00, 0.50, 1.00)),
    (("gpt-4.1-mini",),             ModelPrice("OpenAI", 0.40, 1.60, 0.50, 1.00)),
    (("gpt-4.1",),                  ModelPrice("OpenAI", 2.00, 8.00, 0.50, 1.00)),
    (("gpt-4o-mini",),              ModelPrice("OpenAI", 0.15, 0.60, 0.50, 1.00)),
    # DeepSeek
    (("deepseek-chat",),            ModelPrice("DeepSeek", 0.27, 1.10, 0.10, 1.00)),
    (("deepseek-reasoner",),        ModelPrice("DeepSeek", 0.55, 2.19, 0.10, 1.00)),
    # Google
    (("gemini-3-pro-preview", "gemini-3-pro"), ModelPrice("Google", 2.00, 9.00, 0.20, 1.00)),
    (("gemini-3.1-pro",),           ModelPrice("Google", 4.00, 18.00, 0.25, 1.00)),
    (("gemini-3.1-flash",),         ModelPrice("Google", 0.50, 3.00, 0.25, 1.00)),
    (("gemini-3-flash-preview", "gemini-3-flash"), ModelPrice("Google", 0.50, 3.00, 0.25, 1.00)),
    (("gemini-2.5-pro",),           ModelPrice("Google", 1.25, 10.00, 0.25, 1.00)),
    (("gemini-2.5-flash",),         ModelPrice("Google", 0.30, 2.50, 0.25, 1.00)),
    # Together.ai (no caching assumed)
    (("zai-org/GLM-5.1", "GLM-5.1"), ModelPrice("Together.ai", 1.40, 4.40, 1.00, 1.00)),
    (("MiniMaxAI/MiniMax-M2.7", "MiniMax-M2.7"), ModelPrice("Together.ai", 0.30, 1.20, 1.00, 1.00)),
    (("moonshotai/Kimi-K2.5", "Kimi-K2.5"), ModelPrice("Together.ai", 0.50, 2.80, 1.00, 1.00)),
]


def lookup_price(model: str) -> ModelPrice | None:
    """Return the pricing entry for a model, or None if unknown (e.g. self-hosted)."""
    for prefixes, price in _REGISTRY:
        for p in prefixes:
            if model.startswith(p) or model == p:
                return price
    return None


def compute_cost(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    price: ModelPrice,
) -> dict:
    """Return a cost breakdown dict for a single call or aggregate.

    Formulas:
      regular_input_cost    = prompt_tokens      × input_per_m / 1e6
      cache_read_cost       = cache_read_tokens  × input_per_m × cache_read_mult  / 1e6
      cache_creation_cost   = cache_creation     × input_per_m × cache_write_mult / 1e6
      output_cost           = completion_tokens  × output_per_m / 1e6
    """
    reg = prompt_tokens * price.input_per_m / 1e6
    cr = cache_read_tokens * price.input_per_m * price.cache_read_mult / 1e6
    cw = cache_creation_tokens * price.input_per_m * price.cache_write_mult / 1e6
    out = completion_tokens * price.output_per_m / 1e6
    return {
        "regular_input_usd": round(reg, 6),
        "cache_read_usd": round(cr, 6),
        "cache_creation_usd": round(cw, 6),
        "output_usd": round(out, 6),
        "total_usd": round(reg + cr + cw + out, 6),
    }
