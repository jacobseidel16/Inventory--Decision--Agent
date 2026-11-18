"""Supply Chain Analytics Inventory Decision Agent.

Implements the explicit 7-stage decision pipeline:
    Customer Question
        -> Identify Model
        -> Extract Inputs (full-question semantic fact extraction & disambiguation)
        -> Select Python Tool
        -> Calculate (strictly in Python)
        -> Verify
        -> Explain

Examines the entire customer question — across all sentences, clauses, bullets,
parentheticals, and domain synonyms — extracting both numerical values and their
semantic meanings before deciding whether any parameter is missing.
Never invents missing numerical inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from inventory_tools import (
    TOOL_REGISTRY,
    InventoryValidationError,
    ToolSpec,
)


# ---------------------------------------------------------------------------
# Pipeline Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedFact:
    parameter: str
    value: Any
    meaning: str
    source_text: str


@dataclass
class ModelIdentification:
    model_id: Optional[str]
    display_name: str
    confidence: float
    matched_signals: List[str] = field(default_factory=list)
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    method: str = "deterministic_classifier"


@dataclass
class InputExtraction:
    inputs: Dict[str, Any] = field(default_factory=dict)
    extracted_facts: List[ExtractedFact] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    method: str = "deterministic_semantic_parser"


@dataclass
class ToolSelection:
    model_id: str
    tool_name: str
    display_name: str
    arguments: Dict[str, Any]


@dataclass
class VerificationCheck:
    name: str
    passed: bool
    details: str


@dataclass
class VerificationResult:
    passed: bool
    checks: List[VerificationCheck] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    question: str
    status: str  # "success" | "needs_clarification" | "error"
    identification: ModelIdentification
    extraction: InputExtraction
    tool_selection: Optional[ToolSelection] = None
    calculation: Optional[Dict[str, Any]] = None
    verification: Optional[VerificationResult] = None
    explanation: str = ""
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Numeric, Clause & Semantic Parsing Helpers
# ---------------------------------------------------------------------------

_NUM_PAT = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[kK])?"


def _parse_number(raw: str) -> float:
    cleaned = raw.strip().replace(",", "").replace("$", "").replace("%", "")
    multiplier = 1.0
    if cleaned.lower().endswith("k"):
        multiplier = 1000.0
        cleaned = cleaned[:-1]
    return float(cleaned) * multiplier


def _normalize_rate(value: float, had_percent_sign: bool) -> float:
    if had_percent_sign or value > 1.0:
        return value / 100.0
    return value


def _split_clauses(text: str) -> List[str]:
    """Split a full natural-language prompt into sentences/lines/semicolon clauses
    without breaking decimal numbers or bullet lists."""
    normalized = text.replace("\r\n", "\n").strip()
    raw_parts = re.split(r"(?:\n+|;\s+|(?<=[.!?])\s+)", normalized)
    clauses: List[str] = []
    for part in raw_parts:
        cleaned = part.strip().lstrip("•-*").strip()
        if cleaned:
            clauses.append(cleaned)
    return clauses if clauses else [normalized]


def _search_number_with_match(text: str, patterns: List[str]) -> Tuple[Optional[float], str]:
    """Return (parsed_number, matched_source_text) for the first matching pattern."""
    for pat in patterns:
        match = re.search(pat, text, flags=re.IGNORECASE)
        if match:
            try:
                return _parse_number(match.group(1)), match.group(0).strip()
            except (ValueError, IndexError):
                continue
    return None, ""


def _record_fact(
    inputs: Dict[str, Any],
    facts: List[ExtractedFact],
    parameter: str,
    value: Any,
    meaning: str,
    source_text: str,
    overwrite: bool = False,
) -> None:
    if not overwrite and parameter in inputs:
        return
    inputs[parameter] = value
    # Replace prior fact for the same parameter if overwriting
    if overwrite:
        facts[:] = [f for f in facts if f.parameter != parameter]
    facts.append(
        ExtractedFact(
            parameter=parameter,
            value=value,
            meaning=meaning,
            source_text=source_text,
        )
    )


def _detect_time_unit(text: str, default: str = "day") -> str:
    lower = text.lower()
    if re.search(r"\b(per\s+week|weekly|/\s*week|a\s+week)\b", lower):
        return "week"
    if re.search(r"\b(per\s+month|monthly|/\s*month|a\s+month)\b", lower):
        return "month"
    if re.search(r"\b(per\s+day|daily|/\s*day|a\s+day)\b", lower):
        return "day"
    if re.search(r"\bweeks?\b", lower) and not re.search(r"\bdays?\b", lower):
        return "week"
    return default


def _convert_duration(value: float, from_unit: str, to_unit: str) -> Tuple[float, Optional[str]]:
    u_from = from_unit.lower().rstrip("s")
    u_to = to_unit.lower().rstrip("s")
    if u_from == u_to:
        return value, None

    to_days = {"day": 1.0, "week": 7.0, "month": 30.0, "year": 365.0}
    if u_from in to_days and u_to in to_days:
        converted = value * (to_days[u_from] / to_days[u_to])
        note = f"Converted {value:g} {u_from}(s) to {converted:g} {u_to}(s) to match demand time units."
        return converted, note
    return value, None


# ---------------------------------------------------------------------------
# Ordering Cost Synonym Patterns (shared by EOQ & Quantity Discount)
# ---------------------------------------------------------------------------

_ORDERING_COST_PATTERNS: List[str] = [
    # Explicit ordering/setup cost synonyms:
    # - ordering cost, order cost, setup cost, setup expense, ordering/setup cost,
    #   setup/ordering cost, order placement cost, order processing cost, fixed order cost
    rf"(?:fixed\s+)?(?:ordering\s*/\s*setup|setup\s*/\s*ordering|order\s*/\s*setup|ordering|order|setup|set-up|order\s+placement|order\s+processing)\s+"
    rf"(?:cost|expense|fee|charge)\s*(?:is|are|of|=|:)?\s*\$?\s*({_NUM_PAT})",
    # - cost per order, expense per order, cost of placing an order, cost to place an order
    rf"(?:cost|expense|fee|charge)\s+(?:per\s+order|of\s+placing\s+(?:an?\s+)?order|to\s+place\s+(?:an?\s+)?order|per\s+setup)\s*(?:is|of|=|:)?\s*\$?\s*({_NUM_PAT})",
    # - per-order cost / per order cost
    rf"per[\s-]order\s+(?:cost|expense|fee)\s*(?:is|of|=|:)?\s*\$?\s*({_NUM_PAT})",
    # - each order costs $X / placing an order costs $X / it costs $X to place an order
    rf"(?:each\s+order|placing\s+an?\s+order|order\s+placement)\s+costs?\s*\$\s*({_NUM_PAT})",
    rf"costs?\s*\$\s*({_NUM_PAT})\s*(?:per\s+order|/\s*order|to\s+place\s+an?\s+order|for\s+each\s+order|per\s+setup)",
    # - $X per order / $X/order / $X per setup
    rf"\$\s*({_NUM_PAT})\s*(?:per\s+order|/\s*order|per\s+setup|/\s*setup|to\s+place\s+an?\s+order)",
    # - Symbolic S = $X
    rf"\bS\s*=\s*\$?\s*({_NUM_PAT})",
]


# ---------------------------------------------------------------------------
# Price-Break Extraction for Quantity Discount
# ---------------------------------------------------------------------------

def _extract_unit_purchase_price(text: str) -> Tuple[Optional[float], str]:
    """Extract base/regular unit purchase price (distinct from holding cost & tier prices)."""
    return _search_number_with_match(
        text,
        [
            rf"(?:regular|base|standard|list|normal)\s+(?:unit\s+)?(?:purchase\s+)?(?:price|cost)\s*(?:is|of|=|:)?\s*\$\s*({_NUM_PAT})",
            rf"(?:unit\s+purchase\s+price|purchase\s+price|unit\s+price|unit\s+cost|item\s+cost|acquisition\s+cost)\s*(?:is|of|=|:)?\s*\$\s*({_NUM_PAT})",
            rf"(?:costs?|priced\s+at)\s*\$\s*({_NUM_PAT})\s*(?:per\s+unit|each|/\s*unit)(?!\s*(?:per\s+year|annually|/\s*year|to\s+hold|holding|carrying))",
            rf"\bC\s*=\s*\$?\s*({_NUM_PAT})",
        ],
    )


def _extract_price_breaks(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Extract quantity discount tiers (quantity breaks + unit prices) from natural language."""
    tiers: List[Dict[str, Any]] = []
    assumptions: List[str] = []

    # 1. Range before price: "1-499 units at $20", "500–999 units: $11.50", "500 to 999: $18.50"
    range_before_price = re.finditer(
        rf"\b({_NUM_PAT})\s*(?:-|–|—|to)\s*({_NUM_PAT})\s*(?:units?|items?|pcs?|pieces?)?\s*"
        rf"(?:at|for|costs?|cost|price(?:d)?\s*(?:at)?|unit\s+price\s*(?:is|of)?|is|are|:|=)?\s*\$\s*({_NUM_PAT})",
        text,
        flags=re.IGNORECASE,
    )
    for m in range_before_price:
        tiers.append(
            {
                "min_qty": _parse_number(m.group(1)),
                "max_qty": _parse_number(m.group(2)),
                "unit_cost": _parse_number(m.group(3)),
            }
        )

    # 2. Price before range: "$20 for 1-499 units", "$18 for orders of 500 to 999"
    price_before_range = re.finditer(
        rf"\$\s*({_NUM_PAT})\s*(?:per\s+unit|each|/unit)?\s*(?:for|if\s+ordering|on|when\s+ordering)\s+"
        rf"(?:orders?\s+of\s+)?({_NUM_PAT})\s*(?:-|–|—|to)\s*({_NUM_PAT})\s*(?:units?|items?)?",
        text,
        flags=re.IGNORECASE,
    )
    for m in price_before_range:
        tiers.append(
            {
                "min_qty": _parse_number(m.group(2)),
                "max_qty": _parse_number(m.group(3)),
                "unit_cost": _parse_number(m.group(1)),
            }
        )

    # 3. Open lower bound: "under 500 at $20", "Orders below 500 units: $12.00", "up to 499 at $20"
    open_low_before_price = re.finditer(
        rf"(?:(under|less\s+than|fewer\s+than|below|<)|(up\s+to|<=|≤))\s*({_NUM_PAT})\s*(?:units?|items?|pcs?|pieces?)?\s*"
        rf"(?:at|for|costs?|price(?:d)?\s*(?:at)?|is|:|=)?\s*\$\s*({_NUM_PAT})",
        text,
        flags=re.IGNORECASE,
    )
    for m in open_low_before_price:
        upper = _parse_number(m.group(3))
        max_q = max(1.0, upper - 1.0) if m.group(1) else upper
        tiers.append({"min_qty": 1.0, "max_qty": max_q, "unit_cost": _parse_number(m.group(4))})

    price_before_open_low = re.finditer(
        rf"\$\s*({_NUM_PAT})\s*(?:per\s+unit|each|/unit)?\s*(?:for|if\s+ordering|on|when\s+ordering)\s+"
        rf"(?:orders?\s+of\s+)?(?:(under|less\s+than|fewer\s+than|below|<)|(up\s+to|<=|≤))\s*({_NUM_PAT})",
        text,
        flags=re.IGNORECASE,
    )
    for m in price_before_open_low:
        upper = _parse_number(m.group(4))
        max_q = max(1.0, upper - 1.0) if m.group(2) else upper
        tiers.append({"min_qty": 1.0, "max_qty": max_q, "unit_cost": _parse_number(m.group(1))})

    # 4. Open upper bound:
    #    "1,000+ at $16", "1,000 or more units: $11.00", "1,000 units or more: $11.00", ">= 1,000: $16"
    open_high_before_price = re.finditer(
        rf"(?:(?:at\s+least|>=|≥|over|more\s+than)\s*({_NUM_PAT})|({_NUM_PAT})\s*(?:(?:units?|items?|pcs?|pieces?)\s+)?(?:\+|or\s+more|and\s+up|and\s+above))\s*"
        rf"(?:units?|items?|pcs?|pieces?)?\s*(?:at|for|costs?|price(?:d)?\s*(?:at)?|is|:|=)?\s*\$\s*({_NUM_PAT})",
        text,
        flags=re.IGNORECASE,
    )
    for m in open_high_before_price:
        q_raw = m.group(1) or m.group(2)
        tiers.append({"min_qty": _parse_number(q_raw), "max_qty": None, "unit_cost": _parse_number(m.group(3))})

    price_before_open_high = re.finditer(
        rf"\$\s*({_NUM_PAT})\s*(?:per\s+unit|each|/unit)?\s*(?:for|if\s+ordering|on|when\s+ordering)\s+"
        rf"(?:orders?\s+of\s+)?(?:(?:at\s+least|>=|≥|over|more\s+than)\s*({_NUM_PAT})|({_NUM_PAT})\s*(?:(?:units?|items?|pcs?|pieces?)\s+)?(?:\+|or\s+more|and\s+up|and\s+above))",
        text,
        flags=re.IGNORECASE,
    )
    for m in price_before_open_high:
        q_raw = m.group(2) or m.group(3)
        tiers.append({"min_qty": _parse_number(q_raw), "max_qty": None, "unit_cost": _parse_number(m.group(1))})

    # 5. Explicit quantity/price break points paired with a base unit purchase price:
    #    e.g., "Regular unit purchase price is $25, with a quantity break at 500 units ($23) and 1,000 units ($21)"
    if not tiers:
        base_price, _ = _extract_unit_purchase_price(text)
        break_pts = list(
            re.finditer(
                rf"(?:quantity\s+breaks?|price\s+breaks?|breaks?)\s+(?:at\s+)?({_NUM_PAT})\s*(?:units?|items?)?\s*"
                rf"(?:\(\s*\$\s*({_NUM_PAT})\s*\)|(?:at|for|:|-)\s*\$\s*({_NUM_PAT}))",
                text,
                flags=re.IGNORECASE,
            )
        )
        # Also catch subsequent "and 1,000 units ($21)" pairs in the same sentence
        extra_pts = (
            list(
                re.finditer(
                    rf"(?:and|,)\s*({_NUM_PAT})\s*(?:units?|items?)\s*(?:\(\s*\$\s*({_NUM_PAT})\s*\)|(?:at|for|:)\s*\$\s*({_NUM_PAT}))",
                    text,
                    flags=re.IGNORECASE,
                )
            )
            if break_pts
            else []
        )
        all_pts = []
        for m in break_pts + extra_pts:
            qty = _parse_number(m.group(1))
            price = _parse_number(m.group(2) or m.group(3))
            all_pts.append((qty, price))
        if all_pts:
            unique_pts = sorted(set(all_pts), key=lambda x: x[0])
            if base_price is not None and unique_pts[0][0] > 1.0:
                tiers.append({"min_qty": 1.0, "max_qty": unique_pts[0][0] - 1.0, "unit_cost": base_price})
            for idx, (min_q, price) in enumerate(unique_pts):
                max_q = unique_pts[idx + 1][0] - 1.0 if idx + 1 < len(unique_pts) else None
                tiers.append({"min_qty": min_q, "max_qty": max_q, "unit_cost": price})

    # 6. Base unit purchase price + percentage discounts:
    #    e.g., "Unit purchase price is $50, with 5% discount for 500+ and 10% discount for 1,000+"
    if not tiers:
        base_price, _ = _extract_unit_purchase_price(text)
        if base_price is not None:
            pct_matches = list(
                re.finditer(
                    rf"({_NUM_PAT})\s*%\s*(?:off|discount)\s*(?:for|on|at|when\s+ordering|if\s+ordering)?\s*"
                    rf"(?:orders?\s+of\s+)?(?:at\s+least|>=|over)?\s*({_NUM_PAT})\s*(?:\+|or\s+more|units?)?",
                    text,
                    flags=re.IGNORECASE,
                )
            )
            if pct_matches:
                discount_points = sorted(
                    [(_parse_number(m.group(2)), _parse_number(m.group(1)) / 100.0) for m in pct_matches],
                    key=lambda x: x[0],
                )
                tiers.append({"min_qty": 1.0, "max_qty": discount_points[0][0] - 1.0, "unit_cost": base_price})
                for idx, (min_q, disc) in enumerate(discount_points):
                    max_q = discount_points[idx + 1][0] - 1.0 if idx + 1 < len(discount_points) else None
                    tiers.append(
                        {
                            "min_qty": min_q,
                            "max_qty": max_q,
                            "unit_cost": round(base_price * (1.0 - disc), 6),
                        }
                    )
                assumptions.append(
                    f"Constructed all-units discount tiers from base unit purchase price ${base_price:,.2f} and stated percentage discounts."
                )

    deduped: Dict[Tuple[float, float], Dict[str, Any]] = {}
    for t in tiers:
        key = (t["min_qty"], t["unit_cost"])
        if key not in deduped or (deduped[key]["max_qty"] is None and t["max_qty"] is not None):
            deduped[key] = t

    sorted_tiers = sorted(deduped.values(), key=lambda x: (x["min_qty"], -x["unit_cost"]))
    final_tiers: List[Dict[str, Any]] = []
    seen_min = set()
    for t in sorted_tiers:
        if t["min_qty"] not in seen_min:
            seen_min.add(t["min_qty"])
            final_tiers.append(t)

    for i in range(len(final_tiers) - 1):
        if final_tiers[i]["max_qty"] is None:
            final_tiers[i]["max_qty"] = final_tiers[i + 1]["min_qty"] - 1.0

    return final_tiers, assumptions


# ---------------------------------------------------------------------------
# Main Agent Class
# ---------------------------------------------------------------------------

class InventoryDecisionAgent:
    """Natural-language Supply Chain Analytics Inventory Decision Agent."""

    def __init__(self, use_llm: bool = False, llm_model: str = "gpt-4o-mini") -> None:
        self.use_llm = use_llm and bool(os.getenv("OPENAI_API_KEY"))
        self.llm_model = llm_model

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        override_model: Optional[str] = None,
        override_inputs: Optional[Dict[str, Any]] = None,
        uploaded_df: Any = None,
    ) -> AgentResponse:
        """Execute the full 7-stage pipeline on a natural-language question."""
        cleaned_question = (question or "").strip()
        if not cleaned_question and not (override_model and override_inputs):
            ident = ModelIdentification(
                model_id=None,
                display_name="Unknown",
                confidence=0.0,
                matched_signals=[],
            )
            return AgentResponse(
                question=cleaned_question,
                status="needs_clarification",
                identification=ident,
                extraction=InputExtraction(),
                explanation="Please enter an inventory question describing your demand, costs, lead time, or service level.",
            )

        # Stage 1: Identify Model
        if override_model and override_model in TOOL_REGISTRY:
            spec = TOOL_REGISTRY[override_model]
            identification = ModelIdentification(
                model_id=override_model,
                display_name=spec.display_name,
                confidence=1.0,
                matched_signals=["Explicitly selected by user"],
                candidate_scores={override_model: 1.0},
                method="user_override",
            )
        else:
            identification = self.identify_model(cleaned_question)

        if not identification.model_id:
            return AgentResponse(
                question=cleaned_question,
                status="needs_clarification",
                identification=identification,
                extraction=InputExtraction(),
                explanation=(
                    "I could not confidently identify which inventory model fits your question. "
                    "I support: **1) Economic Order Quantity (EOQ)**, **2) Quantity Discount**, "
                    "**3) Safety Stock & Reorder Point**, **4) Fixed-Period Review**, **5) Newsvendor**, "
                    "and **6) Multicriteria Inventory Classification (A/B/C/D)** from an uploaded CSV. "
                    "Please include details such as demand, ordering/holding costs, price breaks, "
                    "lead time & service level, review period, selling price & salvage value, or ask to classify your uploaded CSV."
                ),
            )

        # Stage 2: Extract Inputs (examines the ENTIRE question before checking missing fields)
        extraction = self.extract_inputs(
            cleaned_question, identification.model_id, uploaded_df=uploaded_df
        )
        if override_inputs:
            merged = dict(extraction.inputs)
            merged.update({k: v for k, v in override_inputs.items() if v is not None})
            extraction.inputs = merged
            extraction.missing_fields = self._find_missing_fields(identification.model_id, merged)

        if extraction.missing_fields:
            spec = TOOL_REGISTRY[identification.model_id]
            missing_str = ", ".join(f"`{m}`" for m in extraction.missing_fields)
            safe_extracted = {
                k: (f"<DataFrame ({len(v)} rows)>" if hasattr(v, "columns") else v)
                for k, v in extraction.inputs.items()
            }
            return AgentResponse(
                question=cleaned_question,
                status="needs_clarification",
                identification=identification,
                extraction=extraction,
                explanation=(
                    f"I identified this as a **{spec.display_name}** problem, but I will not invent "
                    f"missing numerical inputs. Please provide: {missing_str}.\n\n"
                    f"**Extracted so far:** `{json.dumps(safe_extracted)}`"
                ),
            )

        # Stage 3: Select Python Tool
        tool_selection = self.select_tool(identification.model_id, extraction.inputs)

        # Stage 4: Calculate (strictly in Python)
        try:
            calculation = self.calculate(tool_selection)
        except InventoryValidationError as exc:
            return AgentResponse(
                question=cleaned_question,
                status="error",
                identification=identification,
                extraction=extraction,
                tool_selection=tool_selection,
                error_message=str(exc),
                explanation=(
                    f"The Python calculation tool `{tool_selection.tool_name}` rejected the inputs "
                    f"due to a validation error: **{exc}**. Please verify the numeric values."
                ),
            )
        except Exception as exc:  # pragma: no cover
            return AgentResponse(
                question=cleaned_question,
                status="error",
                identification=identification,
                extraction=extraction,
                tool_selection=tool_selection,
                error_message=str(exc),
                explanation=f"Unexpected error during Python calculation: {exc}",
            )

        # Stage 5: Verify
        verification = self.verify(
            model_id=identification.model_id,
            inputs=tool_selection.arguments,
            result=calculation,
        )

        # Stage 6: Explain
        explanation = self.explain(
            question=cleaned_question,
            identification=identification,
            extraction=extraction,
            tool_selection=tool_selection,
            calculation=calculation,
            verification=verification,
        )

        return AgentResponse(
            question=cleaned_question,
            status="success",
            identification=identification,
            extraction=extraction,
            tool_selection=tool_selection,
            calculation=calculation,
            verification=verification,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Stage 1: Identify Model
    # ------------------------------------------------------------------

    def identify_model(self, question: str) -> ModelIdentification:
        """Classify a natural-language question into one of the 5 inventory models."""
        text = question.lower()
        scores: Dict[str, float] = {k: 0.0 for k in TOOL_REGISTRY}
        signals: Dict[str, List[str]] = {k: [] for k in TOOL_REGISTRY}

        rules: Dict[str, List[Tuple[str, float, str]]] = {
            "quantity_discount": [
                (r"\bquantity\s+discount", 6.0, "mentions 'quantity discount'"),
                (r"\bquantity\s+break|\bprice\s+break", 6.0, "mentions 'quantity/price break'"),
                (r"\bvolume\s+discount|\bbulk\s+discount|\btiered\s+pricing", 5.5, "mentions volume/bulk discount"),
                (r"\ball[\s-]units?\s+discount", 6.0, "mentions 'all-units discount'"),
                (r"\bdiscount\b", 3.0, "mentions 'discount'"),
                (rf"\b{_NUM_PAT}\s*(?:-|–|—|to)\s*{_NUM_PAT}[^\n]*\$\s*\d+", 4.5, "contains quantity-range pricing tiers"),
                (rf"{_NUM_PAT}\s*(?:(?:units?|items?)\s+)?(?:\+|or\s+more|and\s+up)[^\n]*\$\s*\d+", 3.5, "contains threshold pricing tier"),
            ],
            "newsvendor": [
                (r"\bnewsvendor\b|\bnewsboy\b", 7.0, "mentions 'newsvendor'"),
                (r"\bsingle[\s-]period\b", 6.0, "mentions 'single-period'"),
                (r"\bcritical\s+fractile|\bcritical\s+ratio", 6.0, "mentions 'critical fractile/ratio'"),
                (r"\bunderage\b|\boverage\b|\bshortage\s+cost|\bexcess\s+cost|\bunderstock|\boverstock", 5.0, "mentions underage/overage or shortage/excess cost"),
                (r"\bsalvage\b|\bmarkdown\b|\bunsold\b|\bleftover\b|\bliquidat|\bclearance\b", 4.5, "mentions salvage/unsold value"),
                (r"\bperishable\b|\bseasonal\b|\bone[\s-]time\s+order|\bholiday\b", 3.5, "mentions seasonal/perishable demand"),
                (r"\bselling\s+price|\bretail\s+price|\bsells?\s+[\w\s]{0,20}for\s*\$", 3.0, "mentions selling/retail price"),
            ],
            "fixed_period_review": [
                (r"\bfixed[\s-]period", 7.0, "mentions 'fixed-period'"),
                (r"\bperiodic\s+review|\bp[\s-]system\b", 7.0, "mentions 'periodic review / P-system'"),
                (r"\border[\s-]up[\s-]to|\bbase[\s-]stock|\btarget\s+inventory", 6.0, "mentions 'order-up-to / target inventory'"),
                (r"\breview\s+period|\breview\s+interval|\border\s+cycle", 6.0, "mentions 'review period'"),
                (r"\b(?:review(?:ed)?|check(?:ed)?|reorder(?:ed)?|order(?:ed)?)\s+(?:inventory\s+|stock\s+)?every\s+\d+", 5.5, "mentions periodic review schedule"),
                (r"\binventory\s+position|\binventory\s+on\s+hand|\bon[\s-]hand\s+inventory|\bon[\s-]order\s+inventory|\bbackorders?\b", 2.5, "mentions inventory position components"),
            ],
            "safety_stock_rop": [
                (r"\bsafety\s+stock\b", 5.5, "mentions 'safety stock'"),
                (r"\breorder\s+point\b|\brop\b", 6.0, "mentions 'reorder point / ROP'"),
                (r"\bcontinuous\s+review|\bq[\s-]system\b", 5.5, "mentions 'continuous review'"),
                (r"\bcycle\s+service\s+level|\bservice\s+level\b|\bstockout\b", 3.0, "mentions service level/stockout"),
                (r"\blead[\s-]time\b", 2.5, "mentions lead time"),
                (r"\bstandard\s+deviation|\bstd\.?\s*dev|\bsigma\b", 2.0, "mentions demand/lead-time uncertainty"),
            ],
            "eoq": [
                (r"\beoq\b|\beconomic\s+order\s+quantity", 6.5, "mentions 'EOQ'"),
                (r"\bordering\s*/\s*setup|\bordering\s+cost|\border\s+cost|\bsetup\s+cost|\bsetup\s+expense|\bcost\s+per\s+order|\bcost\s+of\s+placing\s+an?\s+order|\bper\s+order\b", 3.5, "mentions ordering/setup cost"),
                (r"\bholding\s+cost|\bcarrying\s+cost|\bstorage\s+cost|\bholding\s+expense", 3.0, "mentions holding/carrying cost"),
                (r"\bannual\s+demand\b|\byearly\s+demand\b|\bannual\s+usage\b|\bper\s+year\b|\bannually\b", 2.0, "mentions annual demand"),
                (r"\boptimal\s+order\s+quantity|\bhow\s+much\s+should\s+we\s+order", 1.5, "asks for optimal order quantity"),
            ],
            "inventory_classification": [
                (r"\binventory\s+classification|\bclassify\b|\bclassification\b", 7.0, "mentions inventory classification"),
                (r"\bmulticriteria\b|\bmulti[\s-]criteria\b", 7.0, "mentions multicriteria classification"),
                (r"\babc[\s/-]?d\b|\babcd\b|\ba,\s*b,\s*c,?\s*and\s*d\b|\ba/b/c/d\b|\babc\s+analysis", 6.5, "mentions A/B/C/D classification"),
                (r"\bsales\b.*\bhits\b|\bhits\b.*\bsales\b", 5.5, "mentions Sales and Hits criteria"),
                (r"\buploaded\s+(?:csv|file|data|inventory)\b", 3.0, "references uploaded CSV inventory data"),
            ],
        }

        for model_id, model_rules in rules.items():
            for pattern, weight, desc in model_rules:
                if re.search(pattern, text):
                    scores[model_id] += weight
                    signals[model_id].append(desc)

        parsed_tiers, _ = _extract_price_breaks(question)
        if len(parsed_tiers) >= 2:
            scores["quantity_discount"] += 12.0
            signals["quantity_discount"].append(f"detected {len(parsed_tiers)} price-break tiers")

        if scores["fixed_period_review"] >= 5.0:
            scores["fixed_period_review"] += 2.5

        best_model = max(scores, key=lambda k: scores[k])
        best_score = scores[best_model]

        if best_score < 2.5:
            return ModelIdentification(
                model_id=None,
                display_name="Unidentified",
                confidence=0.0,
                matched_signals=[],
                candidate_scores=scores,
            )

        sorted_scores = sorted(scores.values(), reverse=True)
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        margin = best_score - second_score
        confidence = min(0.99, round(0.55 + 0.05 * best_score + 0.04 * margin, 2))

        return ModelIdentification(
            model_id=best_model,
            display_name=TOOL_REGISTRY[best_model].display_name,
            confidence=confidence,
            matched_signals=signals[best_model],
            candidate_scores={k: round(v, 2) for k, v in scores.items()},
            method="deterministic_classifier",
        )

    # ------------------------------------------------------------------
    # Stage 2: Extract Inputs (Full-Question Semantic Extraction)
    # ------------------------------------------------------------------

    def extract_inputs(
        self, question: str, model_id: str, uploaded_df: Any = None
    ) -> InputExtraction:
        """Scan the entire question, extract all numerical values and their semantic
        meanings, and only then determine whether any required parameter is missing."""
        if model_id == "eoq":
            inputs, facts, assumptions = self._extract_eoq_inputs(question)
        elif model_id == "quantity_discount":
            inputs, facts, assumptions = self._extract_quantity_discount_inputs(question)
        elif model_id == "safety_stock_rop":
            inputs, facts, assumptions = self._extract_safety_stock_inputs(question)
        elif model_id == "fixed_period_review":
            inputs, facts, assumptions = self._extract_fixed_period_inputs(question)
        elif model_id == "newsvendor":
            inputs, facts, assumptions = self._extract_newsvendor_inputs(question)
        elif model_id == "inventory_classification":
            inputs, facts, assumptions = self._extract_classification_inputs(question, uploaded_df)
        else:
            inputs, facts, assumptions = {}, [], []

        method = "deterministic_semantic_parser"
        missing = self._find_missing_fields(model_id, inputs)

        if missing and self.use_llm and model_id != "inventory_classification":
            llm_inputs = self._llm_extract_inputs(question, model_id)
            if llm_inputs:
                for k, v in llm_inputs.items():
                    if k not in inputs and v is not None:
                        _record_fact(inputs, facts, k, v, f"LLM-extracted {k}", "LLM fallback")
                method = "hybrid_deterministic_and_llm"
                missing = self._find_missing_fields(model_id, inputs)

        return InputExtraction(
            inputs=inputs,
            extracted_facts=facts,
            missing_fields=missing,
            assumptions=assumptions,
            method=method,
        )

    def _extract_classification_inputs(
        self, text: str, uploaded_df: Any
    ) -> Tuple[Dict[str, Any], List[ExtractedFact], List[str]]:
        inputs: Dict[str, Any] = {}
        facts: List[ExtractedFact] = []
        assumptions: List[str] = []

        if uploaded_df is not None:
            row_cnt = len(uploaded_df) if hasattr(uploaded_df, "__len__") else "?"
            _record_fact(
                inputs,
                facts,
                "df",
                uploaded_df,
                f"Uploaded inventory DataFrame ({row_cnt} rows)",
                "Uploaded CSV via sidebar Data Upload",
            )

        sw_match = re.search(
            rf"(?:sales\s+weight\s*(?:is|of|=|:)?\s*({_NUM_PAT})\s*(%)?|({_NUM_PAT})\s*(%)\s*(?:weight\s+(?:on|for)\s+)?sales)",
            text,
            flags=re.IGNORECASE,
        )
        if sw_match:
            raw_v = sw_match.group(1) or sw_match.group(3)
            had_pct = bool(sw_match.group(2) or sw_match.group(4))
            _record_fact(
                inputs,
                facts,
                "sales_weight",
                _normalize_rate(_parse_number(raw_v), had_pct),
                "Sales ($) criterion weight",
                sw_match.group(0).strip(),
            )

        hw_match = re.search(
            rf"(?:hits\s+weight\s*(?:is|of|=|:)?\s*({_NUM_PAT})\s*(%)?|({_NUM_PAT})\s*(%)\s*(?:weight\s+(?:on|for)\s+)?hits)",
            text,
            flags=re.IGNORECASE,
        )
        if hw_match:
            raw_v = hw_match.group(1) or hw_match.group(3)
            had_pct = bool(hw_match.group(2) or hw_match.group(4))
            _record_fact(
                inputs,
                facts,
                "hits_weight",
                _normalize_rate(_parse_number(raw_v), had_pct),
                "Hits criterion weight",
                hw_match.group(0).strip(),
            )

        return inputs, facts, assumptions

    def _find_missing_fields(self, model_id: str, inputs: Dict[str, Any]) -> List[str]:
        spec = TOOL_REGISTRY[model_id]
        missing: List[str] = []
        for req in spec.required_params:
            val = inputs.get(req)
            if val is None or (isinstance(val, list) and len(val) == 0):
                missing.append(req)

        if spec.alternative_param_groups:
            group_satisfied = any(
                all(inputs.get(p) is not None for p in group)
                for group in spec.alternative_param_groups
            )
            if not group_satisfied:
                group_labels = [" + ".join(g) for g in spec.alternative_param_groups]
                missing.append(f"either ({' OR '.join(group_labels)})")

        if model_id == "safety_stock_rop":
            has_var = (inputs.get("demand_std") or 0.0) > 0.0 or (inputs.get("lead_time_std") or 0.0) > 0.0
            if has_var and inputs.get("service_level") is None and inputs.get("z_score") is None:
                missing.append("service_level (or z_score)")

        if model_id == "fixed_period_review":
            has_var = (inputs.get("demand_std") or 0.0) > 0.0 or (inputs.get("lead_time_std") or 0.0) > 0.0
            has_explicit_ss = inputs.get("safety_stock") is not None
            if has_var and not has_explicit_ss and inputs.get("service_level") is None and inputs.get("z_score") is None:
                missing.append("service_level (or z_score or explicit safety_stock)")

        if model_id == "eoq" and inputs.get("lead_time_days") is not None and inputs.get("working_days_per_year") is None:
            missing.append("working_days_per_year")

        return missing

    # ------------------------------------------------------------------
    # Model 1: EOQ Extractor
    # ------------------------------------------------------------------

    def _extract_eoq_inputs(self, text: str) -> Tuple[Dict[str, Any], List[ExtractedFact], List[str]]:
        inputs: Dict[str, Any] = {}
        facts: List[ExtractedFact] = []
        assumptions: List[str] = []

        # Working days per year
        working_days, src = _search_number_with_match(
            text,
            [
                rf"({_NUM_PAT})\s*(?:working|business|operating)\s+days(?:\s+(?:per|a|in\s+a)\s+year)?",
                rf"(?:working|business|operating)\s+days\s*(?:per\s+year)?\s*(?:is|are|of|=|:)?\s*({_NUM_PAT})",
            ],
        )
        if working_days is not None:
            _record_fact(inputs, facts, "working_days_per_year", working_days, "Working days per year", src)

        # Annual demand (plus monthly / weekly / daily conversions)
        annual_demand, src = _search_number_with_match(
            text,
            [
                rf"(?:annual|yearly)\s+(?:demand|usage|consumption|sales|volume|requirement)\s*(?:is|are|of|=|:)?\s*({_NUM_PAT})",
                rf"(?:demand|usage|consumption|sales)\s*(?:is|of|=|:)?\s*({_NUM_PAT})\s*(?:units?|items?|pcs?|pieces?)?\s*(?:per\s+year|a\s+year|annually|yearly|/\s*year|/\s*yr)",
                rf"(?:sells?|uses?|requires?|consumes?|needs?)\s*({_NUM_PAT})\s*(?:units?|items?|pcs?|pieces?)?\s*(?:per\s+year|a\s+year|annually|yearly|/\s*year|/\s*yr)",
                rf"({_NUM_PAT})\s*(?:units?|items?|pcs?|pieces?)\s*(?:per\s+year|a\s+year|annually|yearly|/\s*year|/\s*yr)",
                rf"\bD\s*=\s*({_NUM_PAT})",
            ],
        )
        if annual_demand is not None:
            _record_fact(inputs, facts, "annual_demand", annual_demand, "Annual demand (D)", src)
        else:
            monthly, src = _search_number_with_match(
                text,
                [
                    rf"monthly\s+(?:demand|usage|sales)\s*(?:is|of|=|:)?\s*({_NUM_PAT})",
                    rf"(?:demand|usage|sales)\s*(?:is|of|=|:)?\s*({_NUM_PAT})\s*(?:units?|items?)?\s*(?:per\s+month|a\s+month|monthly|/\s*month)",
                    rf"({_NUM_PAT})\s*(?:units?|items?)\s*(?:per\s+month|a\s+month|monthly|/\s*month)",
                ],
            )
            weekly, w_src = _search_number_with_match(
                text,
                [
                    rf"weekly\s+(?:demand|usage|sales)\s*(?:is|of|=|:)?\s*({_NUM_PAT})",
                    rf"(?:demand|usage|sales)\s*(?:is|of|=|:)?\s*({_NUM_PAT})\s*(?:units?|items?)?\s*(?:per\s+week|a\s+week|weekly|/\s*week)",
                    rf"({_NUM_PAT})\s*(?:units?|items?)\s*(?:per\s+week|a\s+week|weekly|/\s*week)",
                ],
            )
            daily, d_src = _search_number_with_match(
                text,
                [
                    rf"daily\s+(?:demand|usage|sales)\s*(?:is|of|=|:)?\s*({_NUM_PAT})",
                    rf"(?:demand|usage|sales)\s*(?:is|of|=|:)?\s*({_NUM_PAT})\s*(?:units?|items?)?\s*(?:per\s+day|a\s+day|daily|/\s*day)",
                    rf"({_NUM_PAT})\s*(?:units?|items?)\s*(?:per\s+day|a\s+day|daily|/\s*day)",
                ],
            )
            if monthly is not None:
                annual_demand = monthly * 12.0
                _record_fact(inputs, facts, "annual_demand", annual_demand, "Annualized from monthly demand", src)
                assumptions.append(f"Converted monthly demand ({monthly:g}/month) to annual demand ({annual_demand:g}/year).")
            elif weekly is not None:
                annual_demand = weekly * 52.0
                _record_fact(inputs, facts, "annual_demand", annual_demand, "Annualized from weekly demand", w_src)
                assumptions.append(f"Converted weekly demand ({weekly:g}/week) to annual demand ({annual_demand:g}/year).")
            elif daily is not None and working_days is not None:
                annual_demand = daily * working_days
                _record_fact(inputs, facts, "annual_demand", annual_demand, "Annualized from daily demand", d_src)
                assumptions.append(
                    f"Converted daily demand ({daily:g}/day) to annual demand ({annual_demand:g}/year using {working_days:g} working days/year)."
                )

        # Ordering / setup cost (comprehensive synonym recognition)
        ordering_cost, src = _search_number_with_match(text, _ORDERING_COST_PATTERNS)
        if ordering_cost is not None:
            _record_fact(inputs, facts, "ordering_cost", ordering_cost, "Ordering / setup cost per order (S)", src)

        # Percentage holding cost rate
        rate_match = re.search(
            rf"(?:percentage\s+|annual\s+)?(?:holding\s*/\s*carrying|carrying\s*/\s*holding|holding|carrying|storage|inventory\s+carrying)\s*"
            rf"(?:cost|expense)?\s*(?:rate|percentage|charge|factor)?\s*(?:is|of|=|:)?\s*({_NUM_PAT})\s*%",
            text,
            flags=re.IGNORECASE,
        )
        if not rate_match:
            rate_match = re.search(
                rf"({_NUM_PAT})\s*%\s*(?:of\s+(?:the\s+)?(?:unit\s+)?(?:purchase\s+)?(?:cost|price)|(?:annual\s+)?(?:holding|carrying|storage)\s*(?:cost|rate|charge)?)",
                text,
                flags=re.IGNORECASE,
            )
        if not rate_match:
            rate_match = re.search(
                rf"(?:percentage\s+(?:holding|carrying)\s+cost|(?:holding|carrying)\s+(?:cost\s+)?rate)\s*(?:is|of|=|:)?\s*(0\.\d+)\b",
                text,
                flags=re.IGNORECASE,
            )
        if rate_match:
            raw_r = rate_match.group(1)
            had_pct = "%" in rate_match.group(0)
            rate_val = _normalize_rate(_parse_number(raw_r), had_pct)
            _record_fact(inputs, facts, "holding_cost_rate", rate_val, "Percentage holding cost rate (h)", rate_match.group(0).strip())

        # Fixed dollar holding cost ($/unit/year)
        holding_cost, src = _search_number_with_match(
            text,
            [
                rf"(?:fixed\s+|annual\s+)?(?:holding\s*/\s*carrying|carrying\s*/\s*holding|holding|carrying|storage)\s+"
                rf"(?:cost|expense|charge)\s*(?:is|of|=|:)?\s*\$\s*({_NUM_PAT})",
                rf"\$\s*({_NUM_PAT})\s*(?:per\s+unit\s+per\s+year|/\s*unit\s*/\s*year|/\s*unit\s*/\s*yr|per\s+unit\s+annually|per\s+unit\s+a\s+year)\s*(?:to\s+hold|to\s+store|holding|carrying)?",
                rf"costs?\s*\$\s*({_NUM_PAT})\s*(?:per\s+unit\s+)?(?:per\s+year|annually)?\s*to\s+(?:hold|carry|store)",
                rf"\bH\s*=\s*\$?\s*({_NUM_PAT})",
            ],
        )
        if holding_cost is not None:
            _record_fact(inputs, facts, "holding_cost", holding_cost, "Fixed annual holding cost per unit (H)", src)

        # Unit purchase price / unit cost
        unit_cost, src = _extract_unit_purchase_price(text)
        if unit_cost is not None:
            _record_fact(inputs, facts, "unit_cost", unit_cost, "Unit purchase price / cost (C)", src)

        # Optional deterministic lead time in days
        lead_time_days, src = _search_number_with_match(
            text,
            [
                rf"lead[\s-]time\s*(?:is|of|=|:)?\s*({_NUM_PAT})\s*days?",
                rf"({_NUM_PAT})[\s-]day\s+lead[\s-]time",
            ],
        )
        if lead_time_days is not None:
            _record_fact(inputs, facts, "lead_time_days", lead_time_days, "Deterministic lead time in days", src)

        return inputs, facts, assumptions

    # ------------------------------------------------------------------
    # Model 2: Quantity Discount Extractor
    # ------------------------------------------------------------------

    def _extract_quantity_discount_inputs(
        self, text: str
    ) -> Tuple[Dict[str, Any], List[ExtractedFact], List[str]]:
        inputs, facts, assumptions = self._extract_eoq_inputs(text)

        # Preserve semantic fact for unit_purchase_price if stated, then remove EOQ-only tool args
        for key in ("unit_cost", "working_days_per_year", "lead_time_days"):
            inputs.pop(key, None)

        # Disambiguate fixed vs percentage holding cost so we never pass both to calculate_quantity_discount()
        if "holding_cost_rate" in inputs and "holding_cost" in inputs:
            if re.search(r"fixed\s+(?:holding|carrying)\s+cost", text, flags=re.IGNORECASE):
                inputs.pop("holding_cost_rate", None)
                facts[:] = [f for f in facts if f.parameter != "holding_cost_rate"]
            else:
                inputs.pop("holding_cost", None)
                facts[:] = [f for f in facts if f.parameter != "holding_cost"]

        tiers, tier_assumptions = _extract_price_breaks(text)
        assumptions.extend(tier_assumptions)
        if tiers:
            tier_summary = "; ".join(
                f"[{t['min_qty']:g}–{t['max_qty']:g} @ ${t['unit_cost']:g}]"
                if t["max_qty"] is not None
                else f"[{t['min_qty']:g}+ @ ${t['unit_cost']:g}]"
                for t in tiers
            )
            _record_fact(
                inputs,
                facts,
                "price_breaks",
                tiers,
                f"Quantity breaks & unit prices ({len(tiers)} tiers)",
                tier_summary,
            )

        return inputs, facts, assumptions

    # ------------------------------------------------------------------
    # Shared Clause-Aware Extractor for Safety Stock/ROP & Periodic Review
    # ------------------------------------------------------------------

    def _extract_common_stochastic_inputs(
        self, text: str
    ) -> Tuple[Dict[str, Any], List[ExtractedFact], List[str], str]:
        """Distinguish average demand, demand std dev, average lead time,
        lead-time std dev, and service level across the ENTIRE question using
        both explicit patterns and clause-level subject tracking."""
        inputs: Dict[str, Any] = {}
        facts: List[ExtractedFact] = []
        assumptions: List[str] = []

        demand_period = _detect_time_unit(text, default="day")
        inputs["demand_period"] = demand_period
        clauses = _split_clauses(text)

        # Pass 1: Explicit labeled lead-time standard deviation across whole text
        lt_std_patterns = [
            rf"(?:lead[\s-]time|delivery[\s-]time|replenishment[\s-]time)\s+(?:standard\s+deviation|std\.?\s*dev\.?|sd|sigma)\s*(?:is|of|=|:)?\s*({_NUM_PAT})\s*(days?|weeks?|months?)?",
            rf"(?:standard\s+deviation|std\.?\s*dev\.?|sd)\s+of\s+(?:the\s+)?(?:lead[\s-]time|delivery[\s-]time)\s*(?:is|of|=|:)?\s*({_NUM_PAT})\s*(days?|weeks?|months?)?",
            rf"(?:sigma_L|σ_L|sigmaL)\s*(?:is|=|:)?\s*({_NUM_PAT})\s*(days?|weeks?|months?)?",
        ]
        for pat in lt_std_patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                val = _parse_number(m.group(1))
                unit = m.group(2) or demand_period
                conv, note = _convert_duration(val, unit, demand_period)
                _record_fact(inputs, facts, "lead_time_std", conv, "Lead-time standard deviation (sigma_L)", m.group(0).strip())
                if note:
                    assumptions.append(f"Lead-time std dev: {note}")
                break

        # Pass 2: Clause-by-clause subject-aware extraction across the ENTIRE question
        for clause in clauses:
            c_lower = clause.lower()
            has_demand_subj = bool(re.search(r"\b(?:demand|sales|usage|consumption)\b", c_lower))
            has_lt_subj = bool(re.search(r"\b(?:lead[\s-]time|delivery\s+time|replenishment\s+time)\b", c_lower))

            # Lead-time clause (handles: "Lead time has a mean of 5 days and a standard deviation of 1.5 days",
            # "Lead time follows N(5, 1.5)", "Lead time is constant at 4 days")
            if has_lt_subj:
                # N(mu_L, sigma_L)
                m_lt_norm = re.search(
                    rf"(?:lead[\s-]time|delivery\s+time)[^.;\n]*?N\(\s*({_NUM_PAT})\s*,\s*({_NUM_PAT})\s*\)",
                    clause,
                    flags=re.IGNORECASE,
                )
                if m_lt_norm:
                    _record_fact(inputs, facts, "lead_time", _parse_number(m_lt_norm.group(1)), "Average lead time (L)", m_lt_norm.group(0))
                    _record_fact(inputs, facts, "lead_time_std", _parse_number(m_lt_norm.group(2)), "Lead-time standard deviation (sigma_L)", m_lt_norm.group(0))

                # Constant/fixed lead time
                if re.search(r"\b(?:constant|fixed|deterministic)\s+lead[\s-]time|lead[\s-]time\s+is\s+(?:constant|fixed)", c_lower):
                    _record_fact(inputs, facts, "lead_time_std", 0.0, "Constant lead time (sigma_L = 0)", clause)

                # Lead time mean in this clause
                if "lead_time" not in inputs:
                    m_lt = re.search(
                        rf"(?:average\s+|mean\s+|expected\s+|constant\s+|fixed\s+)?(?:lead[\s-]time|delivery\s+time|replenishment\s+time)\s*"
                        rf"(?:is\s+normally\s+distributed\s+with\s+(?:a\s+)?(?:mean|average)\s+of|has\s+a\s+(?:mean|average)\s+of|averages?|takes?|is\s+constant\s+at|is|of|=|:)?\s*({_NUM_PAT})\s*(days?|weeks?|months?)?",
                        clause,
                        flags=re.IGNORECASE,
                    )
                    if not m_lt:
                        m_lt = re.search(
                            rf"({_NUM_PAT})[\s-](day|week|month)\s+lead[\s-]time",
                            clause,
                            flags=re.IGNORECASE,
                        )
                    if m_lt:
                        lt_val = _parse_number(m_lt.group(1))
                        lt_unit = m_lt.group(2) or demand_period
                        lt_conv, note = _convert_duration(lt_val, lt_unit, demand_period)
                        _record_fact(inputs, facts, "lead_time", lt_conv, "Average lead time (L)", m_lt.group(0).strip())
                        if note:
                            assumptions.append(f"Lead time: {note}")

                # Lead-time std dev inside a lead-time-only clause (or after "lead time" in a combined clause)
                if "lead_time_std" not in inputs:
                    lt_tail = clause[re.search(r"lead[\s-]time|delivery\s+time", clause, flags=re.IGNORECASE).start():]
                    m_lt_sd = re.search(
                        rf"(?:standard\s+deviation|std\.?\s*dev\.?|sd|sigma)\s*(?:is|of|=|:)?\s*({_NUM_PAT})\s*(days?|weeks?|months?)?",
                        lt_tail,
                        flags=re.IGNORECASE,
                    )
                    if m_lt_sd and (not has_demand_subj or not re.search(r"\bdemand\b", lt_tail[: m_lt_sd.start()], flags=re.IGNORECASE)):
                        sd_val = _parse_number(m_lt_sd.group(1))
                        sd_unit = m_lt_sd.group(2) or demand_period
                        sd_conv, note = _convert_duration(sd_val, sd_unit, demand_period)
                        _record_fact(inputs, facts, "lead_time_std", sd_conv, "Lead-time standard deviation (sigma_L)", m_lt_sd.group(0).strip())
                        if note:
                            assumptions.append(f"Lead-time std dev: {note}")

            # Demand clause (handles: "Demand has a mean of 100 units/day and a standard deviation of 20 units",
            # "Demand is constant at 60 units/day", "Demand ~ N(100, 20)")
            if has_demand_subj:
                m_d_norm = re.search(
                    rf"(?:demand|sales)[^.;\n]*?N\(\s*({_NUM_PAT})\s*,\s*({_NUM_PAT})\s*\)",
                    clause,
                    flags=re.IGNORECASE,
                )
                if m_d_norm:
                    _record_fact(inputs, facts, "avg_demand", _parse_number(m_d_norm.group(1)), "Average demand (d)", m_d_norm.group(0))
                    _record_fact(inputs, facts, "demand_std", _parse_number(m_d_norm.group(2)), "Demand standard deviation (sigma_d)", m_d_norm.group(0))

                if re.search(r"\b(?:constant|fixed|deterministic)\s+(?:daily\s+|weekly\s+)?demand|demand\s+is\s+(?:constant|fixed)", c_lower):
                    _record_fact(inputs, facts, "demand_std", 0.0, "Constant demand (sigma_d = 0)", clause)

                if "avg_demand" not in inputs:
                    d_val, d_src = _search_number_with_match(
                        clause,
                        [
                            rf"(?:average|mean|expected|constant|fixed)\s+(?:daily|weekly|monthly)?\s*(?:demand|usage|sales)\s*(?:is|of|=|:)?\s*({_NUM_PAT})",
                            rf"(?:daily|weekly|monthly)\s+(?:demand|usage|sales)\s*(?:averages?|is\s+constant\s+at|is|of|=|:)?\s*({_NUM_PAT})",
                            rf"(?:demand|usage|sales)\s+(?:has\s+a\s+(?:mean|average)\s+of|is\s+normally\s+distributed\s+with\s+(?:a\s+)?(?:mean|average)\s+of|averages?|is\s+constant\s+at|is|of|=|:)\s*({_NUM_PAT})",
                            rf"({_NUM_PAT})\s*(?:units?|items?)\s*(?:per\s+(?:day|week|month)|/\s*(?:day|week|month)|daily|weekly|monthly)",
                        ],
                    )
                    if d_val is not None:
                        _record_fact(inputs, facts, "avg_demand", d_val, f"Average demand per {demand_period} (d)", d_src)

                if "demand_std" not in inputs:
                    # Scope demand std dev before any "lead time" mention in the same clause
                    lt_pos = re.search(r"lead[\s-]time|delivery\s+time", clause, flags=re.IGNORECASE)
                    demand_scope = clause[: lt_pos.start()] if (lt_pos and lt_pos.start() > 0) else clause
                    sd_val, sd_src = _search_number_with_match(
                        demand_scope,
                        [
                            rf"(?:demand\s+|daily\s+|weekly\s+|monthly\s+)?(?:standard\s+deviation|std\.?\s*dev\.?|sd|sigma)\s*(?:of\s+demand)?\s*(?:is|of|=|:)?\s*({_NUM_PAT})",
                            rf"(?:sigma_d|σ_d|sigmad)\s*(?:is|=|:)?\s*({_NUM_PAT})",
                        ],
                    )
                    if sd_val is not None:
                        _record_fact(inputs, facts, "demand_std", sd_val, f"Demand standard deviation per {demand_period} (sigma_d)", sd_src)

        # Pass 3: Whole-text fallbacks (symbolic or standalone phrasing)
        if "avg_demand" not in inputs:
            d_val, d_src = _search_number_with_match(
                text,
                [
                    rf"({_NUM_PAT})\s*(?:units?|items?)\s*(?:per\s+(?:day|week|month)|/\s*(?:day|week|month)|daily|weekly)",
                    rf"\b(?:d_bar|mu_d|d)\s*=\s*({_NUM_PAT})",
                ],
            )
            if d_val is not None:
                _record_fact(inputs, facts, "avg_demand", d_val, f"Average demand per {demand_period} (d)", d_src)

        if "lead_time" not in inputs:
            lt_val, lt_src = _search_number_with_match(
                text,
                [
                    rf"\b(?:L_bar|mu_L|L)\s*=\s*({_NUM_PAT})",
                ],
            )
            if lt_val is not None:
                _record_fact(inputs, facts, "lead_time", lt_val, f"Average lead time in {demand_period}s (L)", lt_src)

        # Service level (including stockout-risk complement) or explicit z-score
        sl_match = re.search(
            rf"(?:target\s+)?(?:cycle\s+)?(?:service\s+level|csl|in[\s-]stock\s+(?:probability|rate)|fill\s+probability)\s*(?:is|of|=|:)?\s*({_NUM_PAT})\s*(%)?",
            text,
            flags=re.IGNORECASE,
        )
        if not sl_match:
            sl_match = re.search(
                rf"({_NUM_PAT})\s*(%)\s*(?:target\s+)?(?:cycle\s+)?(?:service\s+level|csl|in[\s-]stock)",
                text,
                flags=re.IGNORECASE,
            )
        if sl_match:
            csl_val = _normalize_rate(_parse_number(sl_match.group(1)), bool(sl_match.group(2)))
            _record_fact(inputs, facts, "service_level", csl_val, "Target cycle service level (CSL)", sl_match.group(0).strip())
        else:
            so_match = re.search(
                rf"(?:stockout\s+(?:risk|probability|rate)|probability\s+of\s+(?:a\s+)?stockout)\s*(?:is|of|at\s+most|<=|=|:)?\s*({_NUM_PAT})\s*(%)?",
                text,
                flags=re.IGNORECASE,
            )
            if not so_match:
                so_match = re.search(
                    rf"({_NUM_PAT})\s*(%)\s*stockout\s+(?:risk|probability|rate)",
                    text,
                    flags=re.IGNORECASE,
                )
            if so_match:
                alpha = _normalize_rate(_parse_number(so_match.group(1)), bool(so_match.group(2)))
                csl_val = round(1.0 - alpha, 10)
                _record_fact(inputs, facts, "service_level", csl_val, "Cycle service level derived from stockout risk (1 - alpha)", so_match.group(0).strip())
                assumptions.append(f"Converted stated stockout probability ({alpha * 100.0:g}%) to cycle service level ({csl_val * 100.0:g}%).")

        z_val, z_src = _search_number_with_match(
            text,
            [
                rf"\bz[\s-]?(?:score|value|factor)\s*(?:is|of|=|:)?\s*({_NUM_PAT})",
                rf"\bz\s*=\s*({_NUM_PAT})",
            ],
        )
        if z_val is not None:
            _record_fact(inputs, facts, "z_score", z_val, "Standard normal z-score", z_src)

        return inputs, facts, assumptions, demand_period

    # ------------------------------------------------------------------
    # Model 3: Safety Stock & ROP Extractor
    # ------------------------------------------------------------------

    def _extract_safety_stock_inputs(
        self, text: str
    ) -> Tuple[Dict[str, Any], List[ExtractedFact], List[str]]:
        inputs, facts, assumptions, _ = self._extract_common_stochastic_inputs(text)
        return inputs, facts, assumptions

    # ------------------------------------------------------------------
    # Model 4: Fixed-Period Review Extractor
    # ------------------------------------------------------------------

    def _extract_fixed_period_inputs(
        self, text: str
    ) -> Tuple[Dict[str, Any], List[ExtractedFact], List[str]]:
        inputs, facts, assumptions, demand_period = self._extract_common_stochastic_inputs(text)

        # Review period
        rp_match = re.search(
            rf"(?:review\s+(?:period|interval)|fixed[\s-]period|order\s+cycle|replenishment\s+interval)\s*(?:is|of|=|:)?\s*({_NUM_PAT})\s*(days?|weeks?|months?)?",
            text,
            flags=re.IGNORECASE,
        )
        if not rp_match:
            rp_match = re.search(
                rf"(?:review(?:ed)?|check(?:ed)?|reorder(?:ed)?|order(?:ed)?)\s+(?:inventory\s+|stock\s+)?every\s+({_NUM_PAT})\s*(days?|weeks?|months?)?",
                text,
                flags=re.IGNORECASE,
            )
        if not rp_match:
            rp_match = re.search(
                rf"\bevery\s+({_NUM_PAT})\s*(days?|weeks?|months?)",
                text,
                flags=re.IGNORECASE,
            )
        if not rp_match:
            rp_match = re.search(
                rf"\b(?:T|P)\s*=\s*({_NUM_PAT})\s*(days?|weeks?|months?)?",
                text,
            )
        if rp_match:
            rp_val = _parse_number(rp_match.group(1))
            rp_unit = rp_match.group(2) or demand_period
            rp_conv, note = _convert_duration(rp_val, rp_unit, demand_period)
            _record_fact(inputs, facts, "review_period", rp_conv, f"Review period in {demand_period}s (T)", rp_match.group(0).strip())
            if note:
                assumptions.append(f"Review period: {note}")

        # Explicit safety stock (when provided numerically instead of derived from sigma & CSL)
        ss_val, ss_src = _search_number_with_match(
            text,
            [
                rf"(?:safety\s+stock|buffer\s+stock)\s*(?:is|of|=|:)\s*({_NUM_PAT})\s*(?:units?|items?)?",
                rf"({_NUM_PAT})\s*(?:units?|items?)\s+(?:of\s+)?(?:safety\s+stock|buffer\s+stock)",
                rf"\bSS\s*=\s*({_NUM_PAT})",
            ],
        )
        if ss_val is not None:
            _record_fact(inputs, facts, "safety_stock", ss_val, "Explicit safety stock (SS)", ss_src)

        # Distinguish: inventory position, on-hand inventory, on-order inventory, backorders
        inv_pos, ip_src = _search_number_with_match(
            text,
            [
                rf"(?:current\s+|net\s+)?inventory\s+position\s*(?:is|of|=|:)?\s*({_NUM_PAT})",
                rf"\bIP\s*=\s*({_NUM_PAT})",
            ],
        )
        on_hand, oh_src = _search_number_with_match(
            text,
            [
                rf"(?:on[\s-]hand|physical|current)\s+(?:inventory|stock)\s*(?:is|of|=|:)?\s*({_NUM_PAT})",
                rf"(?:inventory|stock)\s+on[\s-]hand\s*(?:is|of|=|:)?\s*({_NUM_PAT})",
                rf"\bon[\s-]hand\s*(?:is|are|of|=|:)\s*({_NUM_PAT})",
                rf"(?:currently\s+have|have|hold(?:ing)?)\s*({_NUM_PAT})\s*(?:units?|items?)?\s*(?:on[\s-]hand|in\s+stock|in\s+inventory)",
                rf"({_NUM_PAT})\s*(?:units?|items?)\s*(?:on[\s-]hand|in\s+stock)",
                rf"\bOH\s*=\s*({_NUM_PAT})",
            ],
        )
        on_order, oo_src = _search_number_with_match(
            text,
            [
                rf"(?:on[\s-]order|in[\s-]transit|pipeline)\s+(?:inventory|stock|units?)\s*(?:is|are|of|=|:)?\s*({_NUM_PAT})",
                rf"(?:inventory|stock)\s+on[\s-]order\s*(?:is|are|of|=|:)?\s*({_NUM_PAT})",
                rf"\bon[\s-]order\s*(?:is|are|of|=|:)\s*({_NUM_PAT})",
                rf"(?:scheduled\s+receipts?|open\s+orders?)\s*(?:is|are|of|=|:)?\s*({_NUM_PAT})",
                rf"({_NUM_PAT})\s*(?:units?|items?)\s*(?:on[\s-]order|in[\s-]transit|scheduled\s+to\s+arrive)",
                rf"\bOO\s*=\s*({_NUM_PAT})",
            ],
        )
        backorders, bo_src = _search_number_with_match(
            text,
            [
                rf"(?:backorders?|backordered\s+units?|unfilled\s+orders?)\s*(?:is|are|of|=|:)?\s*({_NUM_PAT})",
                rf"({_NUM_PAT})\s*(?:units?|items?)?\s*(?:in\s+|on\s+)?(?:backorders?|backordered|unfilled\s+orders?)",
                rf"\bBO\s*=\s*({_NUM_PAT})",
            ],
        )

        if inv_pos is not None:
            _record_fact(inputs, facts, "inventory_position", inv_pos, "Inventory position (IP)", ip_src)
        if on_hand is not None:
            _record_fact(inputs, facts, "on_hand_inventory", on_hand, "On-hand inventory (OH)", oh_src)
        if on_order is not None:
            _record_fact(inputs, facts, "on_order_inventory", on_order, "On-order inventory / scheduled receipts (OO)", oo_src)
        if backorders is not None:
            _record_fact(inputs, facts, "backorders", backorders, "Backorders (BO)", bo_src)

        return inputs, facts, assumptions

    # ------------------------------------------------------------------
    # Model 5: Newsvendor Extractor
    # ------------------------------------------------------------------

    def _extract_newsvendor_inputs(
        self, text: str
    ) -> Tuple[Dict[str, Any], List[ExtractedFact], List[str]]:
        inputs: Dict[str, Any] = {}
        facts: List[ExtractedFact] = []
        assumptions: List[str] = []

        # Demand distribution recognition
        dist_match = re.search(
            r"\b(normally\s+distributed|normal\s+distribution|normal\s+demand|gaussian|uniform\s+distribution|poisson\s+distribution|triangular\s+distribution|exponential\s+distribution)\b",
            text,
            flags=re.IGNORECASE,
        )
        if dist_match:
            matched_dist = dist_match.group(1).lower()
            if "normal" in matched_dist or "gaussian" in matched_dist:
                dist_name = "normal"
            elif "uniform" in matched_dist:
                dist_name = "uniform"
            elif "poisson" in matched_dist:
                dist_name = "poisson"
            elif "triangular" in matched_dist:
                dist_name = "triangular"
            else:
                dist_name = "exponential"
            _record_fact(inputs, facts, "demand_distribution", dist_name, "Demand distribution", dist_match.group(0).strip())
        elif re.search(rf"\bN\(\s*{_NUM_PAT}\s*,\s*{_NUM_PAT}\s*\)", text):
            _record_fact(inputs, facts, "demand_distribution", "normal", "Demand distribution (Normal notation N(mu, sigma))", "N(mu, sigma)")

        # Mean demand & demand standard deviation
        mean_demand, md_src = _search_number_with_match(
            text,
            [
                rf"(?:mean|average|expected|forecast(?:ed)?)\s+demand\s*(?:is|of|=|:)?\s*({_NUM_PAT})",
                rf"demand\s+(?:is\s+normally\s+distributed|follows\s+a\s+normal\s+distribution)?\s*(?:with\s+)?(?:a\s+)?(?:mean|average)\s*(?:is|of|=|:)?\s*({_NUM_PAT})",
                rf"(?:mean|average)\s*(?:is|of|=|:)?\s*({_NUM_PAT})\s*(?:units?|items?)?\s*(?:and|,)\s*(?:a\s+)?(?:standard\s+deviation|std\.?\s*dev\.?|sd|sigma)",
                rf"N\(\s*({_NUM_PAT})\s*,",
                rf"\b(?:mu|μ)\s*=\s*({_NUM_PAT})",
            ],
        )
        if mean_demand is not None:
            _record_fact(inputs, facts, "mean_demand", mean_demand, "Mean demand (mu)", md_src)

        demand_std, sd_src = _search_number_with_match(
            text,
            [
                rf"(?:demand\s+)?(?:standard\s+deviation|std\.?\s*dev\.?|sd|sigma|σ)\s*(?:of\s+demand)?\s*(?:is|of|=|:)?\s*({_NUM_PAT})",
                rf"N\(\s*{_NUM_PAT}\s*,\s*({_NUM_PAT})\s*\)",
            ],
        )
        if demand_std is not None:
            _record_fact(inputs, facts, "demand_std", demand_std, "Demand standard deviation (sigma)", sd_src)

        # Direct underage / shortage cost & overage / excess cost
        underage, cu_src = _search_number_with_match(
            text,
            [
                rf"(?:underage|shortage|understock(?:ing)?)\s+(?:cost|expense|penalty)\s*(?:is|of|=|:)?\s*\$?\s*({_NUM_PAT})",
                rf"cost\s+of\s+(?:underage|understock(?:ing)?|(?:a\s+)?shortage)\s*(?:is|of|=|:)?\s*\$?\s*({_NUM_PAT})",
                rf"\bC_?u\s*=\s*\$?\s*({_NUM_PAT})",
            ],
        )
        overage, co_src = _search_number_with_match(
            text,
            [
                rf"(?:overage|excess|overstock(?:ing)?)\s+(?:cost|expense|penalty)\s*(?:is|of|=|:)?\s*\$?\s*({_NUM_PAT})",
                rf"cost\s+of\s+(?:overage|overstock(?:ing)?|excess(?:\s+inventory)?)\s*(?:is|of|=|:)?\s*\$?\s*({_NUM_PAT})",
                rf"\bC_?o\s*=\s*\$?\s*({_NUM_PAT})",
            ],
        )
        if underage is not None:
            _record_fact(inputs, facts, "underage_cost", underage, "Underage / shortage cost (Cu)", cu_src)
        if overage is not None:
            _record_fact(inputs, facts, "overage_cost", overage, "Overage / excess cost (Co)", co_src)

        # Selling price
        selling_price, p_src = _search_number_with_match(
            text,
            [
                rf"(?:selling|sale|retail)\s+price\s*(?:is|of|=|:)?\s*\$\s*({_NUM_PAT})",
                rf"revenue\s+per\s+unit\s*(?:is|of|=|:)?\s*\$\s*({_NUM_PAT})",
                rf"\bsells?\s+(?:[\w\s]{{0,25}}?)\s*(?:for|at)\s*\$\s*({_NUM_PAT})",
                rf"(?<!unsold\s)(?<!leftover\s)\bsold\s+(?:for|at)\s*\$\s*({_NUM_PAT})",
                rf"\bp\s*=\s*\$?\s*({_NUM_PAT})",
            ],
        )
        if selling_price is not None:
            _record_fact(inputs, facts, "selling_price", selling_price, "Selling price per unit (p)", p_src)

        # Purchase / unit cost
        unit_cost, c_src = _search_number_with_match(
            text,
            [
                rf"(?:unit\s+purchase|purchase|unit|wholesale|procurement|production|manufacturing)\s+(?:cost|price)\s*(?:is|of|=|:)?\s*\$\s*({_NUM_PAT})",
                rf"cost\s+per\s+unit\s*(?:is|of|=|:)?\s*\$\s*({_NUM_PAT})",
                rf"\b(?:costs(?:\s+us)?|bought|buy(?:s)?|purchase(?:s|d)?|pay(?:s)?)\s+(?:[\w\s]{{0,20}}?)\s*(?:for|at)?\s*\$\s*({_NUM_PAT})",
                rf"\bc\s*=\s*\$?\s*({_NUM_PAT})",
            ],
        )
        if unit_cost is not None:
            _record_fact(inputs, facts, "unit_cost", unit_cost, "Purchase / unit cost (c)", c_src)

        # Salvage value (or disposal cost / explicit zero salvage)
        salvage, v_src = _search_number_with_match(
            text,
            [
                rf"(?:salvage|scrap|liquidation|markdown|clearance|buyback)\s*(?:value|price)\s*(?:is|of|=|:)?\s*\$?\s*({_NUM_PAT})",
                rf"(?:unsold|leftover|remaining)\s+(?:[\w\s]{{0,20}}?)\s*(?:are\s+|can\s+be\s+)?(?:sold|salvaged|discounted|liquidated|returned)\s+(?:for|at)\s*\$\s*({_NUM_PAT})",
                rf"(?:salvaged|discounted|marked\s+down|liquidated|returned)\s+(?:for|at|to)\s*\$\s*({_NUM_PAT})",
                rf"\bv\s*=\s*\$?\s*({_NUM_PAT})",
            ],
        )
        disposal_cost, disp_src = _search_number_with_match(
            text,
            [
                rf"disposal\s+(?:cost|fee)\s*(?:is|of|=|:)?\s*\$\s*({_NUM_PAT})",
                rf"costs?\s*\$\s*({_NUM_PAT})\s*(?:per\s+unit\s+)?to\s+dispose",
            ],
        )
        if salvage is not None:
            _record_fact(inputs, facts, "salvage_value", salvage, "Salvage value per unsold unit (v)", v_src)
        elif disposal_cost is not None:
            _record_fact(inputs, facts, "salvage_value", -disposal_cost, "Negative salvage value (disposal cost)", disp_src)
            assumptions.append(
                f"Represented disposal cost of ${disposal_cost:,.2f}/unit as negative salvage value (-${disposal_cost:,.2f})."
            )
        else:
            zero_salv = re.search(
                r"\b(?:no\s+salvage(?:\s+value)?|zero\s+salvage(?:\s+value)?|worthless|discarded|thrown\s+away|cannot\s+be\s+salvaged)\b",
                text,
                flags=re.IGNORECASE,
            )
            if zero_salv:
                _record_fact(inputs, facts, "salvage_value", 0.0, "Zero salvage value (explicitly stated)", zero_salv.group(0))

        # Optional goodwill / stockout penalty (only when price & cost are used so it doesn't collide with direct shortage_cost)
        if "underage_cost" not in inputs:
            penalty, pen_src = _search_number_with_match(
                text,
                [
                    rf"(?:goodwill|stockout)\s+(?:penalty|cost)\s*(?:is|of|=|:)?\s*\$\s*({_NUM_PAT})",
                    rf"loss\s+of\s+goodwill\s*(?:is|of|=|:)?\s*\$\s*({_NUM_PAT})",
                ],
            )
            if penalty is not None:
                _record_fact(inputs, facts, "stockout_penalty", penalty, "Additional stockout / goodwill penalty (g)", pen_src)

        return inputs, facts, assumptions

    def _llm_extract_inputs(self, question: str, model_id: str) -> Optional[Dict[str, Any]]:
        """Optional OpenAI JSON extraction fallback when OPENAI_API_KEY is available."""
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            return None

        spec = TOOL_REGISTRY[model_id]
        allowed_params = spec.required_params + spec.optional_params
        client = OpenAI()
        prompt = (
            f"Extract numeric parameters for the '{spec.display_name}' inventory model "
            f"from the following user question. Return ONLY valid JSON with keys chosen from "
            f"{allowed_params}. Use decimals for percentages (e.g., 0.95 for 95%). "
            f"NEVER invent missing numbers; omit any field not explicitly stated.\n\nQuestion: {question}"
        )
        try:
            resp = client.chat.completions.create(
                model=self.llm_model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You extract structured parameters for supply chain models. Never calculate or invent numbers."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            return {k: v for k, v in data.items() if k in allowed_params}
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Stage 3: Select Python Tool
    # ------------------------------------------------------------------

    def select_tool(self, model_id: str, extracted_inputs: Dict[str, Any]) -> ToolSelection:
        """Select the exact Python calculation tool and filter valid arguments."""
        spec: ToolSpec = TOOL_REGISTRY[model_id]
        allowed = set(spec.required_params + spec.optional_params)
        arguments = {k: v for k, v in extracted_inputs.items() if k in allowed and v is not None}
        return ToolSelection(
            model_id=model_id,
            tool_name=spec.function_name,
            display_name=spec.display_name,
            arguments=arguments,
        )

    # ------------------------------------------------------------------
    # Stage 4: Calculate (Strictly Python)
    # ------------------------------------------------------------------

    def calculate(self, tool_selection: ToolSelection) -> Dict[str, Any]:
        """Execute the selected Python tool function."""
        spec = TOOL_REGISTRY[tool_selection.model_id]
        return spec.func(**tool_selection.arguments)

    # ------------------------------------------------------------------
    # Stage 5: Verify (reads deterministic Python verification checks)
    # ------------------------------------------------------------------

    def verify(
        self,
        model_id: str,
        inputs: Dict[str, Any],
        result: Dict[str, Any],
    ) -> VerificationResult:
        """Collect the deterministic verification checks produced by the Python tool
        and attach operational advisory warnings."""
        tool_checks = result.get("verification_checks", [])
        checks: List[VerificationCheck] = [
            VerificationCheck(
                name=str(c["name"]),
                passed=bool(c["passed"]),
                details=str(c["details"]),
            )
            for c in tool_checks
        ]
        warnings: List[str] = []

        inter = result["intermediate_calculations"]
        final = result["final_answer"]

        if model_id == "eoq" and final.get("orders_per_year", 0) > 365:
            warnings.append(
                "Optimal policy implies ordering more than once per day; consider practical order batching."
            )
        elif model_id == "fixed_period_review":
            ip = inter.get("inventory_position")
            target = final.get("target_inventory")
            if ip is not None and target is not None and ip > target:
                warnings.append(
                    f"Current inventory position ({ip:,.2f}) exceeds target inventory ({target:,.2f}); no order is needed this cycle."
                )
        elif model_id == "newsvendor" and inter.get("clamped_at_zero"):
            warnings.append(
                "Unclamped optimal quantity was negative due to high demand variance and low critical ratio; clamped Q* to 0."
            )

        all_passed = all(c.passed for c in checks) if checks else False
        return VerificationResult(passed=all_passed, checks=checks, warnings=warnings)

    # ------------------------------------------------------------------
    # Stage 6: Explain (strictly renders deterministic Python tool output)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_value(val: Any) -> str:
        if hasattr(val, "columns") and hasattr(val, "shape"):
            return f"DataFrame ({val.shape[0]} rows x {val.shape[1]} columns)"
        if isinstance(val, float):
            if abs(val - round(val)) < 1e-9:
                return f"{round(val):,}"
            return f"{val:,.4f}".rstrip("0").rstrip(".")
        if isinstance(val, list):
            return json.dumps(val)
        return str(val)

    def explain(
        self,
        question: str,
        identification: ModelIdentification,
        extraction: InputExtraction,
        tool_selection: ToolSelection,
        calculation: Dict[str, Any],
        verification: VerificationResult,
    ) -> str:
        """Render the 8-part student-verifiable solution strictly from Python tool output.

        Never reconstructs or recalculates numerical intermediate steps outside Python.
        """
        model_id = identification.model_id
        inputs_extracted = calculation["inputs_extracted"]
        var_defs = calculation["variable_definitions"]
        formulas = calculation["formula"]
        substituted = calculation["formula_with_substituted_values"]
        inter = calculation["intermediate_calculations"]
        final = calculation["final_answer"]
        assumptions = list(extraction.assumptions) + list(calculation.get("assumptions", []))

        lines: List[str] = [f"## {identification.display_name} — Verified Solution\n"]

        # 1. Inputs Extracted
        lines.append("### 1. Inputs Extracted")
        if extraction.extracted_facts:
            for fact in extraction.extracted_facts:
                lines.append(
                    f"- **`{fact.parameter}`** = `{self._format_value(fact.value)}` — {fact.meaning} *(from: \"{fact.source_text}\")*"
                )
        else:
            for k, v in inputs_extracted.items():
                lines.append(f"- **`{k}`** = `{self._format_value(v)}`")

        # 2. Variable Definitions
        lines.append("\n### 2. Variable Definitions")
        for symbol, meaning in var_defs.items():
            lines.append(f"- **`{symbol}`**: {meaning}")

        # 3. Formula
        lines.append("\n### 3. Formula")
        for expr in formulas:
            lines.append(f"- `{expr}`")

        # 4. Formula with Substituted Values
        lines.append("\n### 4. Formula with Substituted Values")
        for step in substituted:
            lines.append(f"- `{step}`")

        # 5. Intermediate Calculations
        lines.append("\n### 5. Intermediate Calculations")
        if model_id == "quantity_discount" and "tier_evaluations" in inter:
            lines.append(f"- **Holding Cost Mode:** `{inter['holding_cost_mode']}`")
            lines.append("")
            lines.append("| Tier | Qty Range | Unit Price | H_i | Raw EOQ | Status / Feasible Qty | Annual Purchase | Annual Ordering | Annual Holding | Total Annual Cost |")
            lines.append("|---:|---|---:|---:|---:|---|---:|---:|---:|---:|")
            opt_tier = final["optimal_tier"]
            for t in inter["tier_evaluations"]:
                hi = f"{t['max_qty']:,.0f}" if t["max_qty"] is not None else "∞"
                mark = " *(Optimal)*" if t["tier"] == opt_tier else ""
                if t.get("is_feasible_candidate") and t["feasible_order_qty"] is not None:
                    q_label = (
                        f"{t['feasible_order_qty']:,.2f} (EOQ feasible)"
                        if t["eoq_in_range"]
                        else f"{t['feasible_order_qty']:,.2f} (adjusted to min qty)"
                    )
                    lines.append(
                        f"| {t['tier']}{mark} | {t['min_qty']:,.0f}–{hi} | ${t['unit_cost']:,.2f} | "
                        f"${t['holding_cost_per_unit']:,.2f} | {t['raw_eoq']:,.2f} | {q_label} | "
                        f"${t['annual_purchasing_cost']:,.2f} | ${t['annual_ordering_cost']:,.2f} | "
                        f"${t['annual_holding_cost']:,.2f} | ${t['total_annual_cost']:,.2f} |"
                    )
                else:
                    lines.append(
                        f"| {t['tier']} | {t['min_qty']:,.0f}–{hi} | ${t['unit_cost']:,.2f} | "
                        f"${t['holding_cost_per_unit']:,.2f} | {t['raw_eoq']:,.2f} | Infeasible (EOQ > max) | "
                        f"— | — | — | — |"
                    )
        elif model_id == "inventory_classification" and "class_summary" in inter:
            lines.append(f"- **Total Items:** {inter['total_items']:,}")
            lines.append(f"- **Total Sales ($):** ${inter['total_sales']:,.2f}")
            lines.append(f"- **Total Hits:** {self._format_value(inter['total_hits'])}")
            lines.append("")
            lines.append("| Final Inventory Class | Item Count | % of Items | Total Sales ($) | % of Sales | Total Hits | % of Hits |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for row in inter["class_summary"]:
                lines.append(
                    f"| **{row['Final Inventory Class']}** | {row['Item Count']:,} | {row['% of Items']:.2f}% | "
                    f"${row['Total Sales ($)']:,.2f} | {row['% of Sales']:.2f}% | "
                    f"{self._format_value(row['Total Hits'])} | {row['% of Hits']:.2f}% |"
                )
        else:
            for k, v in inter.items():
                if v is not None:
                    lines.append(f"- **`{k}`**: {self._format_value(v)}")

        # 6. Final Answer
        lines.append("\n### 6. Final Answer")
        if model_id == "quantity_discount":
            lines.append(f"- **Optimal Tier:** Tier {final['optimal_tier']}")
            lines.append(f"- **Optimal Order Quantity:** {self._format_value(final['optimal_order_qty'])} units")
            lines.append(f"- **Unit Price:** ${final['optimal_unit_cost']:,.2f}")
            lines.append(f"- **Orders per Year:** {self._format_value(final['orders_per_year'])}")
            lines.append(f"- **Annual Purchase Cost:** ${final['annual_purchasing_cost']:,.2f}")
            lines.append(f"- **Annual Ordering Cost:** ${final['annual_ordering_cost']:,.2f}")
            lines.append(f"- **Annual Holding Cost:** ${final['annual_holding_cost']:,.2f}")
            lines.append(f"- **Total Annual Cost:** ${final['total_annual_cost']:,.2f}")
            if final.get("annual_savings_vs_base_tier") is not None:
                lines.append(f"- **Annual Savings vs. Highest-Price Feasible Tier:** ${final['annual_savings_vs_base_tier']:,.2f}")
        else:
            for k, v in final.items():
                if v is not None:
                    lines.append(f"- **`{k}`**: {self._format_value(v)}")

        # 7. Verification Checks
        passed_count = sum(1 for c in verification.checks if c.passed)
        status_label = "PASSED" if verification.passed else "FAILED"
        lines.append(
            f"\n### 7. Verification Checks ({status_label}: {passed_count}/{len(verification.checks)} passed)"
        )
        for c in verification.checks:
            badge = "PASS" if c.passed else "FAIL"
            lines.append(f"- **[{badge}] {c.name}:** {c.details}")
        for w in verification.warnings:
            lines.append(f"- **Warning:** {w}")

        # 8. Assumptions
        lines.append("\n### 8. Assumptions")
        for a in assumptions:
            lines.append(f"- {a}")

        return "\n".join(lines)
