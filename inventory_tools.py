"""Deterministic Pure-Python Quantitative Tools for Supply Chain Inventory Analytics.

Implements five student-verifiable deterministic inventory optimization tools:
1. calculate_eoq()               - Economic Order Quantity (EOQ)
2. calculate_quantity_discount() - All-Units Quantity Discount
3. calculate_rop()               - Safety Stock and Reorder Point (all 4 regimes)
4. calculate_periodic_review()   - Fixed-Period Review (Order-Up-To / P-system)
5. calculate_newsvendor()        - Single-Period Newsvendor Model

Every tool strictly validates inputs, never invents missing numerical inputs,
computes all numerical steps deterministically in Python, and returns an 8-part
student-verifiable structured dictionary:
    1. `inputs_extracted`
    2. `variable_definitions`
    3. `formula`
    4. `formula_with_substituted_values`
    5. `intermediate_calculations`
    6. `final_answer`
    7. `verification_checks`
    8. `assumptions`
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Optional, Tuple

STANDARD_NORMAL = NormalDist(mu=0.0, sigma=1.0)


class InventoryValidationError(ValueError):
    """Raised when an inventory tool receives missing or mathematically invalid inputs."""


def _validate_finite(name: str, value: Any) -> float:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or math.isnan(value)
        or math.isinf(value)
    ):
        raise InventoryValidationError(f"{name} must be a finite numeric value (got {value!r}).")
    return float(value)


def _validate_positive(name: str, value: Any) -> float:
    val = _validate_finite(name, value)
    if val <= 0.0:
        raise InventoryValidationError(f"{name} must be strictly positive (got {val}).")
    return val


def _validate_non_negative(name: str, value: Any) -> float:
    val = _validate_finite(name, value)
    if val < 0.0:
        raise InventoryValidationError(f"{name} must be non-negative (got {val}).")
    return val


def _validate_probability(name: str, value: Any) -> float:
    val = _validate_finite(name, value)
    if not (0.0 < val < 1.0):
        raise InventoryValidationError(
            f"{name} must be strictly between 0 and 1 (got {val})."
        )
    return val


def _fmt_num(val: float, decimals: int = 4) -> str:
    """Format a float cleanly for substituted formula display."""
    if abs(val - round(val)) < 1e-9:
        return f"{round(val):,}"
    return f"{val:,.{decimals}f}".rstrip("0").rstrip(".")


def _fmt_money(val: float) -> str:
    """Format a currency amount cleanly (preserving sub-cent precision if present)."""
    if abs(val * 100.0 - round(val * 100.0)) < 1e-6:
        return f"${val:,.2f}"
    return f"${val:,.4f}".rstrip("0").rstrip(".")


def _make_check(name: str, passed: bool, details: str) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def standard_normal_z(probability: float) -> float:
    """Return the inverse standard normal CDF (z-score) for a probability in (0, 1)."""
    p = _validate_probability("probability", probability)
    return STANDARD_NORMAL.inv_cdf(p)


def standard_normal_loss(z: float) -> float:
    """Return the standard normal loss function L(z) = phi(z) - z * (1 - Phi(z))."""
    return STANDARD_NORMAL.pdf(z) - z * (1.0 - STANDARD_NORMAL.cdf(z))


def _resolve_z_and_service_level(
    service_level: Optional[float],
    z_score: Optional[float],
) -> Tuple[float, float]:
    """Validate and resolve (service_level, z_score) without inventing defaults."""
    if service_level is None and z_score is None:
        raise InventoryValidationError(
            "Stochastic demand or lead time requires an explicit `service_level` (e.g., 0.95) "
            "or `z_score`; missing numerical inputs will not be invented."
        )
    if service_level is not None and z_score is not None:
        csl = _validate_probability("service_level", service_level)
        z = _validate_finite("z_score", z_score)
        implied_csl = STANDARD_NORMAL.cdf(z)
        if abs(implied_csl - csl) > 1e-3:
            raise InventoryValidationError(
                f"Provided service_level ({csl}) and z_score ({z}, Phi(z)={implied_csl:.4f}) "
                "are inconsistent."
            )
        return csl, z
    if service_level is not None:
        csl = _validate_probability("service_level", service_level)
        return csl, standard_normal_z(csl)
    z = _validate_finite("z_score", z_score)
    return STANDARD_NORMAL.cdf(z), z


# ---------------------------------------------------------------------------
# 1. Economic Order Quantity (EOQ)
# ---------------------------------------------------------------------------

def calculate_eoq(
    annual_demand: float,
    ordering_cost: float,
    holding_cost: Optional[float] = None,
    unit_cost: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
    working_days_per_year: Optional[float] = None,
    lead_time_days: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate the Economic Order Quantity: Q = sqrt(2 * D * S / H)."""
    D = _validate_positive("annual_demand", annual_demand)
    S = _validate_positive("ordering_cost", ordering_cost)

    assumptions: List[str] = [
        "Annual demand (D) is constant, known, and occurs at a uniform rate.",
        "Fixed ordering cost (S) and annual holding cost per unit (H) are constant.",
        "Replenishment arrives instantaneously with no stockouts or backorders.",
    ]

    C: Optional[float] = None
    h_rate: Optional[float] = None
    derived_holding = False

    if holding_cost is not None:
        H = _validate_positive("holding_cost", holding_cost)
        if unit_cost is not None:
            C = _validate_positive("unit_cost", unit_cost)
        if holding_cost_rate is not None:
            h_rate = _validate_positive("holding_cost_rate", holding_cost_rate)
            if C is not None and abs((C * h_rate) - H) > 1e-6:
                raise InventoryValidationError(
                    f"Provided holding_cost ({H}) conflicts with unit_cost * holding_cost_rate "
                    f"({C * h_rate})."
                )
        elif C is not None:
            h_rate = H / C
        assumptions.append(f"Annual holding cost per unit was provided directly as H = {_fmt_money(H)}/unit/year.")
    elif unit_cost is not None and holding_cost_rate is not None:
        C = _validate_positive("unit_cost", unit_cost)
        h_rate = _validate_positive("holding_cost_rate", holding_cost_rate)
        H = C * h_rate
        derived_holding = True
        assumptions.append(
            f"Annual holding cost per unit was calculated from unit cost and rate: "
            f"H = {_fmt_num(h_rate)} * {_fmt_money(C)} = {_fmt_money(H)}/unit/year."
        )
    else:
        raise InventoryValidationError(
            "Provide either `holding_cost` or both `unit_cost` and `holding_cost_rate`."
        )

    days_per_year: Optional[float] = None
    if working_days_per_year is not None:
        days_per_year = _validate_positive("working_days_per_year", working_days_per_year)

    lt_days: Optional[float] = None
    if lead_time_days is not None:
        lt_days = _validate_non_negative("lead_time_days", lead_time_days)
        if days_per_year is None:
            raise InventoryValidationError(
                "`working_days_per_year` must be explicitly provided when `lead_time_days` is specified "
                "so daily demand can be computed without inventing a calendar assumption."
            )

    numerator_2ds = 2.0 * D * S
    ratio_2ds_h = numerator_2ds / H
    eoq = math.sqrt(ratio_2ds_h)

    orders_per_year = D / eoq
    cycle_time_years = eoq / D
    cycle_time_days = (cycle_time_years * days_per_year) if days_per_year is not None else None
    avg_cycle_inventory = eoq / 2.0

    annual_ordering_cost = (D / eoq) * S
    annual_holding_cost = (eoq / 2.0) * H
    total_relevant_cost = annual_ordering_cost + annual_holding_cost

    annual_purchasing_cost = (D * C) if C is not None else None
    total_annual_cost = (
        annual_purchasing_cost + total_relevant_cost
        if annual_purchasing_cost is not None
        else total_relevant_cost
    )
    if C is None:
        assumptions.append(
            "Unit purchase cost was not provided; total annual cost reflects ordering + holding costs only."
        )

    daily_demand: Optional[float] = None
    deterministic_rop: Optional[float] = None
    if lt_days is not None and days_per_year is not None:
        daily_demand = D / days_per_year
        deterministic_rop = daily_demand * lt_days

    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any] = {
        "annual_demand": D,
        "ordering_cost": S,
    }
    if holding_cost is not None:
        inputs_extracted["holding_cost"] = H
    if C is not None:
        inputs_extracted["unit_cost"] = C
    if holding_cost_rate is not None:
        inputs_extracted["holding_cost_rate"] = h_rate
    if days_per_year is not None:
        inputs_extracted["working_days_per_year"] = days_per_year
    if lt_days is not None:
        inputs_extracted["lead_time_days"] = lt_days

    # 2. Variable Definitions
    variable_definitions: Dict[str, str] = {
        "D": "Annual demand (units/year)",
        "S": "Fixed ordering / setup cost per order ($/order)",
        "H": "Annual holding / carrying cost per unit ($/unit/year)",
        "Q*": "Economic Order Quantity (units/order)",
        "N": "Number of orders per year (orders/year)",
        "TC_relevant": "Total relevant annual cost = Annual Ordering Cost + Annual Holding Cost ($/year)",
    }
    if C is not None:
        variable_definitions["C"] = "Unit purchase cost ($/unit)"
    if holding_cost_rate is not None:
        variable_definitions["h"] = "Annual holding cost rate (fraction of unit cost)"
    if days_per_year is not None:
        variable_definitions["W"] = "Working days per year (days/year)"
    if lt_days is not None:
        variable_definitions["L"] = "Deterministic lead time (days)"

    # 3. Formula & 4. Formula with Substituted Values
    formulas: List[str] = []
    substituted: List[str] = []

    if derived_holding and C is not None and h_rate is not None:
        formulas.append("H = h * C")
        substituted.append(f"H = {_fmt_num(h_rate)} * {_fmt_money(C)} = {_fmt_money(H)}/unit/year")

    formulas.extend(
        [
            "Q* = sqrt((2 * D * S) / H)",
            "N = D / Q*",
            "Average Cycle Inventory = Q* / 2",
            "Annual Ordering Cost = (D / Q*) * S",
            "Annual Holding Cost = (Q* / 2) * H",
            "TC_relevant = Annual Ordering Cost + Annual Holding Cost",
        ]
    )
    substituted.extend(
        [
            f"Q* = sqrt((2 * {_fmt_num(D)} * {_fmt_money(S)}) / {_fmt_money(H)}) = sqrt({_fmt_num(numerator_2ds)} / {_fmt_num(H)}) = sqrt({_fmt_num(ratio_2ds_h)}) = {eoq:,.4f} units",
            f"N = {_fmt_num(D)} / {eoq:,.4f} = {orders_per_year:,.4f} orders/year",
            f"Average Cycle Inventory = {eoq:,.4f} / 2 = {avg_cycle_inventory:,.4f} units",
            f"Annual Ordering Cost = ({_fmt_num(D)} / {eoq:,.4f}) * {_fmt_money(S)} = ${annual_ordering_cost:,.2f}",
            f"Annual Holding Cost = ({eoq:,.4f} / 2) * {_fmt_money(H)} = ${annual_holding_cost:,.2f}",
            f"TC_relevant = ${annual_ordering_cost:,.2f} + ${annual_holding_cost:,.2f} = ${total_relevant_cost:,.2f}",
        ]
    )

    if days_per_year is not None and cycle_time_days is not None:
        formulas.append("Time Between Orders (days) = (Q* / D) * W")
        substituted.append(
            f"Time Between Orders = ({eoq:,.4f} / {_fmt_num(D)}) * {_fmt_num(days_per_year)} = {cycle_time_days:,.2f} days"
        )
    if C is not None and annual_purchasing_cost is not None:
        formulas.append("Total Annual Cost = (D * C) + TC_relevant")
        substituted.append(
            f"Total Annual Cost = ({_fmt_num(D)} * {_fmt_money(C)}) + ${total_relevant_cost:,.2f} = "
            f"${annual_purchasing_cost:,.2f} + ${total_relevant_cost:,.2f} = ${total_annual_cost:,.2f}"
        )
    if lt_days is not None and days_per_year is not None and daily_demand is not None and deterministic_rop is not None:
        formulas.append("Deterministic ROP = (D / W) * L")
        substituted.append(
            f"Deterministic ROP = ({_fmt_num(D)} / {_fmt_num(days_per_year)}) * {_fmt_num(lt_days)} = "
            f"{daily_demand:,.4f} * {_fmt_num(lt_days)} = {deterministic_rop:,.4f} units"
        )

    # 7. Verification Checks (computed deterministically in Python)
    rel_gap = abs(annual_ordering_cost - annual_holding_cost) / max(total_relevant_cost, 1e-12)
    tc_low = (D / (0.99 * eoq)) * S + (0.99 * eoq / 2.0) * H
    tc_high = (D / (1.01 * eoq)) * S + (1.01 * eoq / 2.0) * H
    verification_checks = [
        _make_check(
            "Positive & Finite EOQ",
            math.isfinite(eoq) and eoq > 0.0 and total_relevant_cost > 0.0,
            f"Q* = {eoq:,.4f} units and TC_relevant = ${total_relevant_cost:,.2f} are finite and positive.",
        ),
        _make_check(
            "Ordering-Holding Cost Balance",
            rel_gap < 1e-6,
            f"Annual ordering cost (${annual_ordering_cost:,.2f}) equals annual holding cost (${annual_holding_cost:,.2f}); relative diff = {rel_gap:.2e}.",
        ),
        _make_check(
            "Local Minimum Optimality (±1% Perturbation)",
            (tc_low > total_relevant_cost) and (tc_high > total_relevant_cost),
            f"TC(0.99 Q*) = ${tc_low:,.4f} > TC(Q*) = ${total_relevant_cost:,.4f} < TC(1.01 Q*) = ${tc_high:,.4f}.",
        ),
    ]

    return {
        "model": "eoq",
        "inputs_extracted": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formula_with_substituted_values": substituted,
        "intermediate_calculations": {
            "holding_cost_per_unit": H,
            "implied_holding_cost_rate": h_rate,
            "numerator_2DS": numerator_2ds,
            "ratio_2DS_over_H": ratio_2ds_h,
            "orders_per_year": orders_per_year,
            "cycle_time_years": cycle_time_years,
            "cycle_time_days": cycle_time_days,
            "average_cycle_inventory": avg_cycle_inventory,
            "annual_ordering_cost": annual_ordering_cost,
            "annual_holding_cost": annual_holding_cost,
            "annual_purchasing_cost": annual_purchasing_cost,
            "daily_demand": daily_demand,
        },
        "final_answer": {
            "eoq": eoq,
            "orders_per_year": orders_per_year,
            "total_relevant_annual_cost": total_relevant_cost,
            "total_annual_cost": total_annual_cost,
            "deterministic_reorder_point": deterministic_rop,
        },
        "verification_checks": verification_checks,
        "assumptions": assumptions,
    }


# ---------------------------------------------------------------------------
# 2. Quantity Discount (All-Units)
# ---------------------------------------------------------------------------

def calculate_quantity_discount(
    annual_demand: float,
    ordering_cost: float,
    price_breaks: List[Dict[str, Any]],
    holding_cost_rate: Optional[float] = None,
    holding_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate all-units quantity discount tiers and compare total annual cost."""
    D = _validate_positive("annual_demand", annual_demand)
    S = _validate_positive("ordering_cost", ordering_cost)

    if holding_cost_rate is None and holding_cost is None:
        raise InventoryValidationError(
            "Provide either `holding_cost_rate` (percentage of unit price) or fixed `holding_cost`."
        )
    if holding_cost_rate is not None and holding_cost is not None:
        raise InventoryValidationError(
            "Provide only one of `holding_cost_rate` or fixed `holding_cost`, not both."
        )

    if holding_cost_rate is not None:
        h_rate = _validate_positive("holding_cost_rate", holding_cost_rate)
        fixed_H = None
        holding_mode = "percentage_of_unit_cost"
    else:
        h_rate = None
        fixed_H = _validate_positive("holding_cost", holding_cost)
        holding_mode = "fixed_per_unit"

    if not price_breaks or not isinstance(price_breaks, list):
        raise InventoryValidationError("price_breaks must be a non-empty list of tier dictionaries.")

    normalized_tiers: List[Dict[str, Any]] = []
    for idx, tier in enumerate(price_breaks):
        if not isinstance(tier, dict):
            raise InventoryValidationError(f"price_breaks[{idx}] must be a dictionary.")
        if "min_qty" not in tier or "unit_cost" not in tier:
            raise InventoryValidationError(
                f"price_breaks[{idx}] must contain both `min_qty` and `unit_cost`."
            )
        min_qty = _validate_non_negative(f"price_breaks[{idx}].min_qty", tier["min_qty"])
        max_qty_raw = tier.get("max_qty")
        max_qty = (
            _validate_positive(f"price_breaks[{idx}].max_qty", max_qty_raw)
            if max_qty_raw is not None
            else None
        )
        unit_cost = _validate_positive(f"price_breaks[{idx}].unit_cost", tier["unit_cost"])
        if max_qty is not None and max_qty < min_qty:
            raise InventoryValidationError(
                f"price_breaks[{idx}] has max_qty ({max_qty}) < min_qty ({min_qty})."
            )
        normalized_tiers.append(
            {"min_qty": min_qty, "max_qty": max_qty, "unit_cost": unit_cost}
        )

    normalized_tiers.sort(key=lambda t: t["min_qty"])

    for i in range(len(normalized_tiers) - 1):
        curr_max = normalized_tiers[i]["max_qty"]
        next_min = normalized_tiers[i + 1]["min_qty"]
        if curr_max is None:
            if next_min <= normalized_tiers[i]["min_qty"]:
                raise InventoryValidationError(
                    f"Duplicate or invalid min_qty between tier {i + 1} and tier {i + 2}."
                )
            normalized_tiers[i]["max_qty"] = next_min - 1.0
        elif curr_max >= next_min:
            raise InventoryValidationError(
                f"Overlapping price breaks: tier {i + 1} max_qty ({curr_max}) "
                f">= tier {i + 2} min_qty ({next_min})."
            )

    evaluated_tiers: List[Dict[str, Any]] = []
    substituted: List[str] = []

    for idx, tier in enumerate(normalized_tiers, start=1):
        C_i = tier["unit_cost"]
        H_i = (h_rate * C_i) if h_rate is not None else fixed_H
        raw_eoq = math.sqrt((2.0 * D * S) / H_i)

        min_q = tier["min_qty"]
        effective_min_q = max(min_q, 1.0)
        max_q = tier["max_qty"]
        next_min_q = normalized_tiers[idx]["min_qty"] if idx < len(normalized_tiers) else None

        max_str = _fmt_num(max_q) if max_q is not None else "infinity"
        if h_rate is not None:
            h_expr = f"H_{idx} = {_fmt_num(h_rate)} * {_fmt_money(C_i)} = {_fmt_money(H_i)}"
        else:
            h_expr = f"H_{idx} = {_fmt_money(H_i)} (fixed)"

        above_upper = (max_q is not None and raw_eoq > max_q + 1e-9) or (
            next_min_q is not None and raw_eoq >= next_min_q - 1e-9
        )

        if raw_eoq < effective_min_q - 1e-9:
            # EOQ is below this tier's minimum qualifying quantity -> evaluate min_qty
            feasible_qty: Optional[float] = effective_min_q
            eoq_in_range = False
            is_feasible_candidate = True
            adjustment = "adjusted_up_to_min_qty"
        elif above_upper:
            # Standard all-units model: EOQ exceeds this tier's upper bound (qualifies for a higher tier)
            feasible_qty = None
            eoq_in_range = False
            is_feasible_candidate = False
            adjustment = "infeasible_above_max_qty"
        else:
            feasible_qty = raw_eoq
            eoq_in_range = True
            is_feasible_candidate = True
            adjustment = "none"

        if is_feasible_candidate and feasible_qty is not None:
            purchasing_cost: Optional[float] = D * C_i
            ordering_cost_annual: Optional[float] = (D / feasible_qty) * S
            holding_cost_annual: Optional[float] = (feasible_qty / 2.0) * H_i
            total_annual_cost: Optional[float] = purchasing_cost + ordering_cost_annual + holding_cost_annual
            orders_yr: Optional[float] = D / feasible_qty

            substituted.append(
                f"Tier {idx} (Qty {_fmt_num(min_q)}-{max_str}, C_{idx} = {_fmt_money(C_i)}): "
                f"{h_expr}; Q_{idx} = sqrt((2 * {_fmt_num(D)} * {_fmt_money(S)}) / {_fmt_money(H_i)}) = {raw_eoq:,.4f} "
                f"-> Feasible Candidate Q_{idx}* = {feasible_qty:,.4f} ({adjustment})"
            )
            substituted.append(
                f"Tier {idx} TC_{idx} = ({_fmt_num(D)} * {_fmt_money(C_i)}) + ({_fmt_num(D)} / {feasible_qty:,.4f}) * {_fmt_money(S)} "
                f"+ ({feasible_qty:,.4f} / 2) * {_fmt_money(H_i)} = ${purchasing_cost:,.2f} + ${ordering_cost_annual:,.2f} "
                f"+ ${holding_cost_annual:,.2f} = ${total_annual_cost:,.2f}"
            )
        else:
            purchasing_cost = None
            ordering_cost_annual = None
            holding_cost_annual = None
            total_annual_cost = None
            orders_yr = None

            substituted.append(
                f"Tier {idx} (Qty {_fmt_num(min_q)}-{max_str}, C_{idx} = {_fmt_money(C_i)}): "
                f"{h_expr}; Q_{idx} = sqrt((2 * {_fmt_num(D)} * {_fmt_money(S)}) / {_fmt_money(H_i)}) = {raw_eoq:,.4f} "
                f"-> Infeasible for Tier {idx} (EOQ exceeds upper limit {max_str}; qualifies for a higher-quantity tier)"
            )

        evaluated_tiers.append(
            {
                "tier": idx,
                "min_qty": min_q,
                "max_qty": max_q,
                "unit_cost": C_i,
                "holding_cost_per_unit": H_i,
                "raw_eoq": raw_eoq,
                "feasible_order_qty": feasible_qty,
                "eoq_in_range": eoq_in_range,
                "is_feasible_candidate": is_feasible_candidate,
                "adjustment": adjustment,
                "orders_per_year": orders_yr,
                "annual_purchasing_cost": purchasing_cost,
                "annual_ordering_cost": ordering_cost_annual,
                "annual_holding_cost": holding_cost_annual,
                "total_annual_cost": total_annual_cost,
            }
        )

    feasible_candidates = [t for t in evaluated_tiers if t["is_feasible_candidate"]]
    if not feasible_candidates:
        # Defensive fallback only if every tier was bounded below its EOQ
        last_tier = evaluated_tiers[-1]
        fallback_q = last_tier["max_qty"] or max(last_tier["min_qty"], 1.0)
        C_last = last_tier["unit_cost"]
        H_last = last_tier["holding_cost_per_unit"]
        p_cost = D * C_last
        o_cost = (D / fallback_q) * S
        h_cost = (fallback_q / 2.0) * H_last
        last_tier.update(
            {
                "feasible_order_qty": fallback_q,
                "is_feasible_candidate": True,
                "adjustment": "clamped_down_to_max_qty",
                "orders_per_year": D / fallback_q,
                "annual_purchasing_cost": p_cost,
                "annual_ordering_cost": o_cost,
                "annual_holding_cost": h_cost,
                "total_annual_cost": p_cost + o_cost + h_cost,
            }
        )
        feasible_candidates = [last_tier]

    best_tier = min(feasible_candidates, key=lambda t: t["total_annual_cost"])
    base_feasible_tier = feasible_candidates[0]
    savings_vs_base = base_feasible_tier["total_annual_cost"] - best_tier["total_annual_cost"]

    tier_costs_str = ", ".join(
        f"TC_{t['tier']}=${t['total_annual_cost']:,.2f}" for t in feasible_candidates
    )
    substituted.append(
        f"Optimal Selection = min({tier_costs_str}) "
        f"-> Tier {best_tier['tier']} with Q* = {best_tier['feasible_order_qty']:,.4f} units "
        f"at {_fmt_money(best_tier['unit_cost'])}/unit and TC* = ${best_tier['total_annual_cost']:,.2f}"
    )

    inputs_extracted: Dict[str, Any] = {
        "annual_demand": D,
        "ordering_cost": S,
        "price_breaks": normalized_tiers,
    }
    if h_rate is not None:
        inputs_extracted["holding_cost_rate"] = h_rate
    else:
        inputs_extracted["holding_cost"] = fixed_H

    variable_definitions: Dict[str, str] = {
        "D": "Annual demand (units/year)",
        "S": "Fixed ordering / setup cost per order ($/order)",
        "C_i": "Unit purchase price in discount tier i ($/unit)",
        "H_i": "Annual holding cost per unit in tier i ($/unit/year)",
        "Q_i": "Unconstrained EOQ for tier i (units)",
        "Q_i*": "Feasible candidate order quantity within tier i's quantity break range (units)",
        "TC_i": "Total annual cost for tier i = Purchasing + Ordering + Holding ($/year)",
    }
    if h_rate is not None:
        variable_definitions["h"] = "Annual percentage holding cost rate (fraction of unit price)"
    else:
        variable_definitions["H"] = "Fixed annual holding cost per unit ($/unit/year)"

    formulas = [
        "H_i = h * C_i" if h_rate is not None else "H_i = H (fixed across all tiers)",
        "Q_i = sqrt((2 * D * S) / H_i)",
        "Feasibility check: if min_qty_i <= Q_i <= max_qty_i, Q_i* = Q_i; if Q_i < min_qty_i, Q_i* = min_qty_i; if Q_i > max_qty_i, tier EOQ is infeasible",
        "TC_i = (D * C_i) + (D / Q_i*) * S + (Q_i* / 2) * H_i",
        "Optimal Policy = argmin_i TC_i across all feasible candidates",
    ]

    # Verification Checks
    q_opt = best_tier["feasible_order_qty"]
    best_next_min = (
        normalized_tiers[best_tier["tier"]]["min_qty"]
        if best_tier["tier"] < len(normalized_tiers)
        else None
    )
    in_bounds = (
        q_opt >= max(best_tier["min_qty"], 1.0) - 1e-9
        and (best_tier["max_qty"] is None or q_opt <= best_tier["max_qty"] + 1e-9)
        and (best_next_min is None or q_opt < best_next_min - 1e-9)
    )
    all_decomp_ok = all(
        abs(
            (t["annual_purchasing_cost"] + t["annual_ordering_cost"] + t["annual_holding_cost"])
            - t["total_annual_cost"]
        )
        < 1e-6
        for t in feasible_candidates
    )
    min_total = min(t["total_annual_cost"] for t in feasible_candidates)
    verification_checks = [
        _make_check(
            "Selected Quantity Within Tier Bounds",
            in_bounds,
            f"Tier {best_tier['tier']} bounds [{_fmt_num(best_tier['min_qty'])}, "
            f"{_fmt_num(best_tier['max_qty']) if best_tier['max_qty'] is not None else 'infinity'}], Q* = {q_opt:,.4f}.",
        ),
        _make_check(
            "Cost Decomposition Identity Across All Feasible Candidates",
            all_decomp_ok,
            f"Purchasing + Ordering + Holding equals Total Annual Cost for all {len(feasible_candidates)} feasible candidate(s).",
        ),
        _make_check(
            "Global Minimum Across All Feasible Candidates",
            abs(best_tier["total_annual_cost"] - min_total) < 1e-6,
            f"Selected Tier {best_tier['tier']} cost (${best_tier['total_annual_cost']:,.2f}) equals global minimum (${min_total:,.2f}).",
        ),
    ]

    assumptions = [
        "All-units quantity discount model: the discounted unit price applies to every unit in the order.",
        "Annual demand (D) and fixed ordering cost (S) are constant and known, with no stockouts.",
        (
            f"Holding cost is a fixed percentage ({h_rate * 100.0:.2f}%) of unit price: H_i = h * C_i."
            if h_rate is not None
            else f"Holding cost is fixed at H = {_fmt_money(fixed_H)}/unit/year across all discount tiers."
        ),
    ]

    return {
        "model": "quantity_discount",
        "inputs_extracted": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formula_with_substituted_values": substituted,
        "intermediate_calculations": {
            "holding_cost_mode": holding_mode,
            "tier_evaluations": evaluated_tiers,
        },
        "final_answer": {
            "optimal_tier": best_tier["tier"],
            "optimal_order_qty": best_tier["feasible_order_qty"],
            "optimal_unit_cost": best_tier["unit_cost"],
            "optimal_holding_cost_per_unit": best_tier["holding_cost_per_unit"],
            "orders_per_year": best_tier["orders_per_year"],
            "annual_purchasing_cost": best_tier["annual_purchasing_cost"],
            "annual_ordering_cost": best_tier["annual_ordering_cost"],
            "annual_holding_cost": best_tier["annual_holding_cost"],
            "total_annual_cost": best_tier["total_annual_cost"],
            "annual_savings_vs_base_tier": savings_vs_base,
        },
        "verification_checks": verification_checks,
        "assumptions": assumptions,
    }


# ---------------------------------------------------------------------------
# 3. Safety Stock and Reorder Point (Continuous Review)
# ---------------------------------------------------------------------------

def calculate_rop(
    avg_demand: float,
    lead_time: float,
    demand_std: Optional[float] = None,
    lead_time_std: Optional[float] = None,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    demand_period: str = "day",
) -> Dict[str, Any]:
    """Calculate Safety Stock (SS) and Reorder Point (ROP) across all 4 variability regimes."""
    d = _validate_positive("avg_demand", avg_demand)
    L = _validate_positive("lead_time", lead_time)
    sigma_d = 0.0 if demand_std is None else _validate_non_negative("demand_std", demand_std)
    sigma_L = 0.0 if lead_time_std is None else _validate_non_negative("lead_time_std", lead_time_std)

    demand_var_flag = sigma_d > 0.0
    lt_var_flag = sigma_L > 0.0

    expected_demand_lt = d * L
    demand_variance_term = L * (sigma_d ** 2)
    lead_time_variance_term = (d ** 2) * (sigma_L ** 2)
    variance_lt = demand_variance_term + lead_time_variance_term

    assumptions: List[str] = [
        f"Continuous-review inventory policy with consistent time units ({demand_period}).",
    ]
    formulas: List[str] = ["mu_LT = d * L"]
    substituted: List[str] = [f"mu_LT = {_fmt_num(d)} * {_fmt_num(L)} = {expected_demand_lt:,.4f} units"]

    csl: Optional[float] = None
    z: Optional[float] = None

    if not demand_var_flag and not lt_var_flag:
        case = "constant_demand_constant_lead_time"
        sigma_lt = 0.0
        if service_level is not None or z_score is not None:
            csl, z = _resolve_z_and_service_level(service_level, z_score)
        safety_stock = 0.0
        rop = expected_demand_lt
        stockout_prob = 0.0
        formulas.extend(["sigma_LT = 0", "SS = 0", "ROP = d * L"])
        substituted.extend(
            [
                "sigma_LT = 0 units",
                "SS = 0 units",
                f"ROP = {_fmt_num(d)} * {_fmt_num(L)} = {rop:,.4f} units",
            ]
        )
        assumptions.append(
            "Case 1 — Constant demand and constant lead time (sigma_d = 0, sigma_L = 0): "
            "no uncertainty exists during lead time, so safety stock is 0."
        )
    else:
        csl, z = _resolve_z_and_service_level(service_level, z_score)
        stockout_prob = 1.0 - csl

        if demand_var_flag and not lt_var_flag:
            case = "variable_demand_constant_lead_time"
            sigma_lt = sigma_d * math.sqrt(L)
            formulas.extend(["sigma_LT = sigma_d * sqrt(L)", "z = Phi^-1(CSL)", "SS = z * sigma_LT", "ROP = mu_LT + SS"])
            substituted.append(
                f"sigma_LT = {_fmt_num(sigma_d)} * sqrt({_fmt_num(L)}) = {_fmt_num(sigma_d)} * {math.sqrt(L):.4f} = {sigma_lt:,.4f} units"
            )
            assumptions.append(
                "Case 2 — Variable demand and constant lead time (sigma_d > 0, sigma_L = 0): "
                "period demands are independent and Normally distributed."
            )
        elif not demand_var_flag and lt_var_flag:
            case = "constant_demand_variable_lead_time"
            sigma_lt = d * sigma_L
            formulas.extend(["sigma_LT = d * sigma_L", "z = Phi^-1(CSL)", "SS = z * sigma_LT", "ROP = mu_LT + SS"])
            substituted.append(
                f"sigma_LT = {_fmt_num(d)} * {_fmt_num(sigma_L)} = {sigma_lt:,.4f} units"
            )
            assumptions.append(
                "Case 3 — Constant demand and variable lead time (sigma_d = 0, sigma_L > 0): "
                "lead time is Normally distributed around its mean."
            )
        else:
            case = "variable_demand_variable_lead_time"
            sigma_lt = math.sqrt(variance_lt)
            formulas.extend(
                [
                    "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)",
                    "z = Phi^-1(CSL)",
                    "SS = z * sigma_LT",
                    "ROP = mu_LT + SS",
                ]
            )
            substituted.append(
                f"sigma_LT = sqrt({_fmt_num(L)} * {_fmt_num(sigma_d)}^2 + {_fmt_num(d)}^2 * {_fmt_num(sigma_L)}^2) = "
                f"sqrt({demand_variance_term:,.4f} + {lead_time_variance_term:,.4f}) = sqrt({variance_lt:,.4f}) = {sigma_lt:,.4f} units"
            )
            assumptions.append(
                "Case 4 — Variable demand and variable lead time (sigma_d > 0, sigma_L > 0): "
                "demand and lead time are statistically independent and lead-time demand is approximated as Normal."
            )

        safety_stock = z * sigma_lt
        rop = expected_demand_lt + safety_stock
        substituted.extend(
            [
                f"z = Phi^-1({csl:.4f}) = {z:.4f}",
                f"SS = {z:.4f} * {sigma_lt:,.4f} = {safety_stock:,.4f} units",
                f"ROP = {expected_demand_lt:,.4f} + {safety_stock:,.4f} = {rop:,.4f} units",
            ]
        )

    inputs_extracted: Dict[str, Any] = {
        "avg_demand": d,
        "lead_time": L,
        "demand_std": sigma_d,
        "lead_time_std": sigma_L,
        "demand_period": demand_period,
    }
    if service_level is not None:
        inputs_extracted["service_level"] = csl
    if z_score is not None:
        inputs_extracted["z_score"] = z

    variable_definitions: Dict[str, str] = {
        "d": f"Average demand per {demand_period} (units/{demand_period})",
        "L": f"Average lead time ({demand_period}s)",
        "sigma_d": f"Demand standard deviation per {demand_period} (units/{demand_period})",
        "sigma_L": f"Lead-time standard deviation ({demand_period}s)",
        "mu_LT": "Expected demand during lead time (units)",
        "sigma_LT": "Standard deviation of demand during lead time (units)",
        "SS": "Safety stock buffer (units)",
        "ROP": "Reorder point triggering replenishment (units)",
    }
    if csl is not None:
        variable_definitions["CSL"] = "Target cycle service level (in-stock probability during lead time)"
    if z is not None:
        variable_definitions["z"] = "Standard normal z-score for target CSL"

    # Verification Checks
    general_sigma = math.sqrt(L * (sigma_d ** 2) + (d ** 2) * (sigma_L ** 2))
    verification_checks = [
        _make_check(
            "Lead-Time Demand Std. Dev. Formula",
            abs(general_sigma - sigma_lt) < 1e-9,
            f"sqrt(L*sigma_d^2 + d^2*sigma_L^2) = {general_sigma:,.6f} matches sigma_LT = {sigma_lt:,.6f}.",
        ),
        _make_check(
            "ROP = Expected Lead-Time Demand + Safety Stock",
            abs((expected_demand_lt + safety_stock) - rop) < 1e-6,
            f"{expected_demand_lt:,.4f} + {safety_stock:,.4f} = {rop:,.4f} units.",
        ),
    ]
    if csl is not None and z is not None:
        cdf_z = STANDARD_NORMAL.cdf(z)
        verification_checks.append(
            _make_check(
                "Normal CDF Round-Trip Accuracy",
                abs(cdf_z - csl) < 1e-6,
                f"Phi({z:.4f}) = {cdf_z:.6f} matches target CSL = {csl:.6f}.",
            )
        )
        if csl >= 0.5:
            verification_checks.append(
                _make_check(
                    "Non-Negative Safety Buffer",
                    safety_stock >= -1e-9,
                    f"Safety stock = {safety_stock:,.4f} units (>= 0 for CSL >= 50%).",
                )
            )

    return {
        "model": "safety_stock_rop",
        "inputs_extracted": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formula_with_substituted_values": substituted,
        "intermediate_calculations": {
            "case": case,
            "expected_demand_during_lead_time": expected_demand_lt,
            "demand_variance_component": demand_variance_term,
            "lead_time_variance_component": lead_time_variance_term,
            "lead_time_demand_variance": variance_lt,
            "sigma_lt": sigma_lt,
            "service_level": csl,
            "z_score": z,
            "stockout_probability": stockout_prob,
        },
        "final_answer": {
            "case": case,
            "safety_stock": safety_stock,
            "reorder_point": rop,
        },
        "verification_checks": verification_checks,
        "assumptions": assumptions,
    }


# ---------------------------------------------------------------------------
# 4. Fixed-Period Review (Periodic Review / Order-Up-To Policy)
# ---------------------------------------------------------------------------

def calculate_periodic_review(
    avg_demand: float,
    review_period: float,
    lead_time: float,
    demand_std: Optional[float] = None,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    safety_stock: Optional[float] = None,
    inventory_position: Optional[float] = None,
    on_hand_inventory: Optional[float] = None,
    on_order_inventory: Optional[float] = None,
    backorders: Optional[float] = None,
    lead_time_std: Optional[float] = None,
    demand_period: str = "day",
) -> Dict[str, Any]:
    """Calculate Fixed-Period Review (Order-Up-To / P-system) inventory policy."""
    d = _validate_positive("avg_demand", avg_demand)
    T = _validate_positive("review_period", review_period)
    L = _validate_non_negative("lead_time", lead_time)
    sigma_d = 0.0 if demand_std is None else _validate_non_negative("demand_std", demand_std)
    sigma_L = 0.0 if lead_time_std is None else _validate_non_negative("lead_time_std", lead_time_std)

    protection_period = T + L
    expected_demand_pp = d * protection_period
    variance_pp = (protection_period * (sigma_d ** 2)) + ((d ** 2) * (sigma_L ** 2))
    sigma_pp = math.sqrt(variance_pp)

    assumptions: List[str] = [
        f"Periodic-review (P-system) policy reviewed every {_fmt_num(T)} {demand_period}(s) with lead time {_fmt_num(L)} {demand_period}(s).",
        f"Protection period spans review period + lead time: T + L = {_fmt_num(protection_period)} {demand_period}(s).",
    ]
    formulas: List[str] = [
        "Protection Period (P) = T + L",
        "Demand during Protection Period (mu_{T+L}) = d * (T + L)",
    ]
    substituted: List[str] = [
        f"P = {_fmt_num(T)} + {_fmt_num(L)} = {_fmt_num(protection_period)} {demand_period}(s)",
        f"mu_{{T+L}} = {_fmt_num(d)} * {_fmt_num(protection_period)} = {expected_demand_pp:,.4f} units",
    ]

    csl: Optional[float] = None
    z: Optional[float] = None

    if safety_stock is not None:
        ss_val = _validate_non_negative("safety_stock", safety_stock)
        if service_level is not None or z_score is not None:
            csl, z = _resolve_z_and_service_level(service_level, z_score)
            if sigma_pp > 0.0 and abs((z * sigma_pp) - ss_val) > max(1.0, 0.05 * ss_val):
                raise InventoryValidationError(
                    f"Provided safety_stock ({ss_val}) conflicts with computed z * sigma_(T+L) ({z * sigma_pp:.2f})."
                )
        formulas.append("SS = given safety stock")
        substituted.append(f"SS = {ss_val:,.4f} units (provided directly)")
        assumptions.append(f"Safety stock was provided directly as SS = {ss_val:,.2f} units.")
    elif sigma_pp == 0.0:
        if service_level is not None or z_score is not None:
            csl, z = _resolve_z_and_service_level(service_level, z_score)
        ss_val = 0.0
        formulas.extend(["sigma_{T+L} = 0", "SS = 0"])
        substituted.extend(["sigma_{T+L} = 0 units", "SS = 0 units"])
        assumptions.append("Demand and lead time are deterministic (zero variance); safety stock is 0.")
    else:
        csl, z = _resolve_z_and_service_level(service_level, z_score)
        ss_val = z * sigma_pp
        if sigma_L > 0.0:
            formulas.extend(["sigma_{T+L} = sqrt((T + L) * sigma_d^2 + d^2 * sigma_L^2)", "z = Phi^-1(CSL)", "SS = z * sigma_{T+L}"])
            substituted.append(
                f"sigma_{{T+L}} = sqrt({_fmt_num(protection_period)} * {_fmt_num(sigma_d)}^2 + {_fmt_num(d)}^2 * {_fmt_num(sigma_L)}^2) = {sigma_pp:,.4f} units"
            )
            assumptions.append(
                "Demand and lead time are independent; demand across the protection period is approximated as Normal."
            )
        else:
            formulas.extend(["sigma_{T+L} = sigma_d * sqrt(T + L)", "z = Phi^-1(CSL)", "SS = z * sigma_{T+L}"])
            substituted.append(
                f"sigma_{{T+L}} = {_fmt_num(sigma_d)} * sqrt({_fmt_num(protection_period)}) = {_fmt_num(sigma_d)} * {math.sqrt(protection_period):.4f} = {sigma_pp:,.4f} units"
            )
            assumptions.append(
                "Period demands are independent and Normally distributed with constant lead time."
            )
        substituted.extend(
            [
                f"z = Phi^-1({csl:.4f}) = {z:.4f}",
                f"SS = {z:.4f} * {sigma_pp:,.4f} = {ss_val:,.4f} units",
            ]
        )

    target_inventory = expected_demand_pp + ss_val
    formulas.append("Target Inventory = mu_{T+L} + SS")
    substituted.append(f"Target Inventory = {expected_demand_pp:,.4f} + {ss_val:,.4f} = {target_inventory:,.4f} units")

    resolved_ip: Optional[float] = None
    order_quantity: Optional[float] = None

    if inventory_position is not None:
        resolved_ip = _validate_finite("inventory_position", inventory_position)
        if on_hand_inventory is not None:
            oh = _validate_non_negative("on_hand_inventory", on_hand_inventory)
            oo = _validate_non_negative("on_order_inventory", on_order_inventory) if on_order_inventory is not None else 0.0
            bo = _validate_non_negative("backorders", backorders) if backorders is not None else 0.0
            comp_ip = oh + oo - bo
            if abs(comp_ip - resolved_ip) > 1e-6:
                raise InventoryValidationError(
                    f"Provided inventory_position ({resolved_ip}) conflicts with "
                    f"on_hand + on_order - backorders ({comp_ip})."
                )
        elif on_order_inventory is not None or backorders is not None:
            raise InventoryValidationError(
                "Provide `inventory_position` alone, or provide `on_hand_inventory` with its components."
            )
        order_quantity = max(0.0, target_inventory - resolved_ip)
        formulas.append("Order Quantity (Q) = max(0, Target Inventory - IP)")
        substituted.append(
            f"Q = max(0, {target_inventory:,.4f} - {_fmt_num(resolved_ip)}) = {order_quantity:,.4f} units"
        )
    elif on_hand_inventory is not None:
        oh = _validate_non_negative("on_hand_inventory", on_hand_inventory)
        oo = _validate_non_negative("on_order_inventory", on_order_inventory) if on_order_inventory is not None else 0.0
        bo = _validate_non_negative("backorders", backorders) if backorders is not None else 0.0
        resolved_ip = oh + oo - bo
        order_quantity = max(0.0, target_inventory - resolved_ip)
        formulas.extend(
            [
                "IP = On-Hand + On-Order - Backorders",
                "Order Quantity (Q) = max(0, Target Inventory - IP)",
            ]
        )
        substituted.extend(
            [
                f"IP = {_fmt_num(oh)} + {_fmt_num(oo)} - {_fmt_num(bo)} = {resolved_ip:,.4f} units",
                f"Q = max(0, {target_inventory:,.4f} - {resolved_ip:,.4f}) = {order_quantity:,.4f} units",
            ]
        )
        omitted = []
        if on_order_inventory is None:
            omitted.append("on_order_inventory = 0")
        if backorders is None:
            omitted.append("backorders = 0")
        if omitted:
            assumptions.append(
                f"Inventory position computed from provided on-hand inventory with unmentioned components absent ({', '.join(omitted)})."
            )
    elif on_order_inventory is not None or backorders is not None:
        raise InventoryValidationError(
            "`on_hand_inventory` (or `inventory_position`) must be provided to compute order quantity."
        )
    else:
        assumptions.append(
            "Inventory position was not provided; order quantity cannot be computed without inventing inventory levels."
        )

    inputs_extracted: Dict[str, Any] = {
        "avg_demand": d,
        "review_period": T,
        "lead_time": L,
        "demand_std": sigma_d,
        "lead_time_std": sigma_L,
        "demand_period": demand_period,
    }
    if service_level is not None:
        inputs_extracted["service_level"] = csl
    if z_score is not None:
        inputs_extracted["z_score"] = z
    if safety_stock is not None:
        inputs_extracted["safety_stock"] = ss_val
    if inventory_position is not None:
        inputs_extracted["inventory_position"] = resolved_ip
    if on_hand_inventory is not None:
        inputs_extracted["on_hand_inventory"] = float(on_hand_inventory)
    if on_order_inventory is not None:
        inputs_extracted["on_order_inventory"] = float(on_order_inventory)
    if backorders is not None:
        inputs_extracted["backorders"] = float(backorders)

    variable_definitions: Dict[str, str] = {
        "d": f"Average demand per {demand_period} (units/{demand_period})",
        "T": f"Review period ({demand_period}s)",
        "L": f"Replenishment lead time ({demand_period}s)",
        "P": f"Protection period = T + L ({demand_period}s)",
        "mu_{T+L}": "Expected demand during the protection period (units)",
        "sigma_{T+L}": "Demand standard deviation during the protection period (units)",
        "SS": "Safety stock covering the protection period (units)",
        "Target Inventory": "Order-up-to target inventory level (units)",
    }
    if csl is not None:
        variable_definitions["CSL"] = "Target cycle service level"
    if z is not None:
        variable_definitions["z"] = "Standard normal z-score for target CSL"
    if resolved_ip is not None:
        variable_definitions["IP"] = "Current inventory position = On-Hand + On-Order - Backorders (units)"
        variable_definitions["Q"] = "Recommended replenishment order quantity (units)"

    # Verification Checks
    verification_checks = [
        _make_check(
            "Protection Period Identity (T + L)",
            abs(protection_period - (T + L)) < 1e-9,
            f"T ({_fmt_num(T)}) + L ({_fmt_num(L)}) = {_fmt_num(protection_period)}.",
        ),
        _make_check(
            "Target Inventory Identity (mu_{T+L} + SS)",
            abs((expected_demand_pp + ss_val) - target_inventory) < 1e-6,
            f"{expected_demand_pp:,.4f} + {ss_val:,.4f} = {target_inventory:,.4f} units.",
        ),
    ]
    if csl is not None and z is not None:
        cdf_z = STANDARD_NORMAL.cdf(z)
        verification_checks.append(
            _make_check(
                "Normal CDF Round-Trip Accuracy",
                abs(cdf_z - csl) < 1e-6,
                f"Phi({z:.4f}) = {cdf_z:.6f} matches target CSL = {csl:.6f}.",
            )
        )
    if resolved_ip is not None and order_quantity is not None:
        consistent = (order_quantity >= 0.0) and (
            abs((resolved_ip + order_quantity) - target_inventory) < 1e-6
            if resolved_ip <= target_inventory
            else order_quantity == 0.0
        )
        verification_checks.append(
            _make_check(
                "Order Quantity Consistency",
                consistent,
                f"max(0, Target {target_inventory:,.4f} - IP {resolved_ip:,.4f}) = {order_quantity:,.4f} units.",
            )
        )

    return {
        "model": "fixed_period_review",
        "inputs_extracted": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formula_with_substituted_values": substituted,
        "intermediate_calculations": {
            "protection_period": protection_period,
            "expected_demand_protection_period": expected_demand_pp,
            "protection_period_variance": variance_pp,
            "sigma_protection_period": sigma_pp,
            "service_level": csl,
            "z_score": z,
            "inventory_position": resolved_ip,
        },
        "final_answer": {
            "protection_period": protection_period,
            "safety_stock": ss_val,
            "target_inventory": target_inventory,
            "order_quantity": order_quantity,
        },
        "verification_checks": verification_checks,
        "assumptions": assumptions,
    }


# ---------------------------------------------------------------------------
# 5. Newsvendor (Single-Period Inventory Model)
# ---------------------------------------------------------------------------

def calculate_newsvendor(
    mean_demand: float,
    demand_std: float,
    selling_price: Optional[float] = None,
    unit_cost: Optional[float] = None,
    salvage_value: Optional[float] = None,
    stockout_penalty: Optional[float] = None,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
    demand_distribution: str = "normal",
) -> Dict[str, Any]:
    """Calculate underage cost (Cu), overage cost (Co), critical ratio (CR),
    and recommended order quantity (Q*) for the single-period Newsvendor model."""
    mu = _validate_positive("mean_demand", mean_demand)
    sigma = _validate_positive("demand_std", demand_std)
    dist_norm = (demand_distribution or "normal").strip().lower()
    if dist_norm not in {"normal", "gaussian"}:
        raise InventoryValidationError(
            f"Unsupported demand_distribution {demand_distribution!r}; "
            "calculate_newsvendor() currently supports the Normal distribution."
        )

    assumptions: List[str] = [
        "Single selling period with one upfront order and no mid-period replenishment.",
        f"Period demand follows a Normal distribution N(mu={_fmt_num(mu)}, sigma={_fmt_num(sigma)}).",
        "Underage cost (Cu) and overage cost (Co) are linear per unit.",
    ]

    p: Optional[float] = None
    c: Optional[float] = None
    v: Optional[float] = None
    g: float = 0.0
    derived_from_price_cost = False

    has_direct = underage_cost is not None or overage_cost is not None
    has_economic = selling_price is not None or unit_cost is not None or salvage_value is not None

    formulas: List[str] = []
    substituted: List[str] = []

    if underage_cost is not None and overage_cost is not None:
        Cu = _validate_positive("underage_cost", underage_cost)
        Co = _validate_positive("overage_cost", overage_cost)
        if selling_price is not None:
            p = _validate_positive("selling_price", selling_price)
        if unit_cost is not None:
            c = _validate_positive("unit_cost", unit_cost)
        if salvage_value is not None:
            v = _validate_finite("salvage_value", salvage_value)
        if stockout_penalty is not None:
            g = _validate_non_negative("stockout_penalty", stockout_penalty)
        assumptions.append(
            f"Underage cost (Cu = {_fmt_money(Cu)}) and overage cost (Co = {_fmt_money(Co)}) were provided directly."
        )
    elif has_direct:
        raise InventoryValidationError(
            "Both `underage_cost` and `overage_cost` must be provided when specifying direct Newsvendor costs."
        )
    elif selling_price is not None and unit_cost is not None:
        if salvage_value is None:
            raise InventoryValidationError(
                "`salvage_value` must be explicitly provided when using `selling_price` and `unit_cost` "
                "(pass 0.0 if unsold units have zero salvage value; missing numerical inputs are not invented)."
            )
        p = _validate_positive("selling_price", selling_price)
        c = _validate_positive("unit_cost", unit_cost)
        v = _validate_finite("salvage_value", salvage_value)
        derived_from_price_cost = True
        if stockout_penalty is not None:
            g = _validate_non_negative("stockout_penalty", stockout_penalty)
            Cu = (p - c) + g
            formulas.append("Cu = p - c + g")
            substituted.append(f"Cu = {_fmt_money(p)} - {_fmt_money(c)} + {_fmt_money(g)} = {_fmt_money(Cu)}/unit")
            assumptions.append(
                f"Computed Cu = p - c + g and Co = c - v using stated goodwill/stockout penalty g = {_fmt_money(g)}."
            )
        else:
            Cu = p - c
            formulas.append("Cu = p - c")
            substituted.append(f"Cu = {_fmt_money(p)} - {_fmt_money(c)} = {_fmt_money(Cu)}/unit")
            assumptions.append("No extra stockout goodwill penalty specified; underage cost equals lost margin (p - c).")
        Co = c - v
        formulas.append("Co = c - v")
        substituted.append(f"Co = {_fmt_money(c)} - {_fmt_money(v)} = {_fmt_money(Co)}/unit")
        if Cu <= 0.0:
            raise InventoryValidationError(
                f"Underage cost Cu = (selling_price - unit_cost + stockout_penalty) must be > 0 (got {Cu})."
            )
        if Co <= 0.0:
            raise InventoryValidationError(
                f"Overage cost Co = (unit_cost - salvage_value) must be > 0 (got {Co})."
            )
    elif has_economic:
        raise InventoryValidationError(
            "Provide all three of `selling_price`, `unit_cost`, and `salvage_value`, "
            "or provide `underage_cost` and `overage_cost` directly."
        )
    else:
        raise InventoryValidationError(
            "Provide either (`underage_cost`, `overage_cost`) or "
            "(`selling_price`, `unit_cost`, `salvage_value`)."
        )

    critical_ratio = Cu / (Cu + Co)
    z_star = standard_normal_z(critical_ratio)
    unclamped_q = mu + z_star * sigma
    recommended_qty = max(0.0, unclamped_q)
    clamped_at_zero = unclamped_q < 0.0

    effective_z = (recommended_qty - mu) / sigma
    loss_val = standard_normal_loss(effective_z)
    expected_shortage = sigma * loss_val
    expected_sales = mu - expected_shortage
    expected_leftover = recommended_qty - expected_sales
    expected_mismatch_cost = (Cu * expected_shortage) + (Co * expected_leftover)
    fill_rate = expected_sales / mu
    in_stock_prob = STANDARD_NORMAL.cdf(effective_z)

    expected_profit: Optional[float] = None
    if p is not None and c is not None and v is not None:
        expected_profit = (
            p * expected_sales
            + v * expected_leftover
            - c * recommended_qty
            - g * expected_shortage
        )

    formulas.extend(
        [
            "CR = Cu / (Cu + Co)",
            "z* = Phi^-1(CR)",
            "Q* = max(0, mu + z* * sigma)",
            "Expected Shortage = sigma * L(z*), where L(z) = phi(z) - z * (1 - Phi(z))",
            "Expected Sales = mu - Expected Shortage",
            "Expected Leftover = Q* - Expected Sales",
            "Expected Mismatch Cost = Cu * Expected Shortage + Co * Expected Leftover",
        ]
    )
    substituted.extend(
        [
            f"CR = {_fmt_money(Cu)} / ({_fmt_money(Cu)} + {_fmt_money(Co)}) = {_fmt_num(Cu)} / {_fmt_num(Cu + Co)} = {critical_ratio:.6f} ({critical_ratio * 100.0:.2f}%)",
            f"z* = Phi^-1({critical_ratio:.6f}) = {z_star:.4f}",
            f"Q* = max(0, {_fmt_num(mu)} + ({z_star:.4f} * {_fmt_num(sigma)})) = {recommended_qty:,.4f} units",
            f"Expected Shortage = {_fmt_num(sigma)} * {loss_val:.6f} = {expected_shortage:,.4f} units",
            f"Expected Sales = {_fmt_num(mu)} - {expected_shortage:,.4f} = {expected_sales:,.4f} units",
            f"Expected Leftover = {recommended_qty:,.4f} - {expected_sales:,.4f} = {expected_leftover:,.4f} units",
            f"Expected Mismatch Cost = ({_fmt_money(Cu)} * {expected_shortage:,.4f}) + ({_fmt_money(Co)} * {expected_leftover:,.4f}) = ${expected_mismatch_cost:,.2f}",
        ]
    )
    if expected_profit is not None and p is not None and c is not None and v is not None:
        formulas.append("Expected Profit = p * E[Sales] + v * E[Leftover] - c * Q* - g * E[Shortage]")
        substituted.append(
            f"Expected Profit = ({_fmt_money(p)} * {expected_sales:,.4f}) + ({_fmt_money(v)} * {expected_leftover:,.4f}) "
            f"- ({_fmt_money(c)} * {recommended_qty:,.4f}) - ({_fmt_money(g)} * {expected_shortage:,.4f}) = ${expected_profit:,.2f}"
        )

    inputs_extracted: Dict[str, Any] = {
        "mean_demand": mu,
        "demand_std": sigma,
        "demand_distribution": "normal",
    }
    if underage_cost is not None:
        inputs_extracted["underage_cost"] = Cu
    if overage_cost is not None:
        inputs_extracted["overage_cost"] = Co
    if p is not None:
        inputs_extracted["selling_price"] = p
    if c is not None:
        inputs_extracted["unit_cost"] = c
    if v is not None:
        inputs_extracted["salvage_value"] = v
    if stockout_penalty is not None:
        inputs_extracted["stockout_penalty"] = g

    variable_definitions: Dict[str, str] = {
        "mu": "Mean single-period demand (units)",
        "sigma": "Standard deviation of single-period demand (units)",
        "Cu": "Underage / shortage cost per unit short ($/unit)",
        "Co": "Overage / excess cost per unsold unit ($/unit)",
        "CR": "Critical ratio = Cu / (Cu + Co) (optimal in-stock probability)",
        "z*": "Standard normal z-score corresponding to CR",
        "Q*": "Recommended single-period order quantity (units)",
    }
    if p is not None:
        variable_definitions["p"] = "Selling price per unit ($/unit)"
    if c is not None:
        variable_definitions["c"] = "Purchase / unit cost ($/unit)"
    if v is not None:
        variable_definitions["v"] = "Salvage value per unsold unit ($/unit; negative if disposal cost)"
    if stockout_penalty is not None:
        variable_definitions["g"] = "Additional goodwill / stockout penalty per unit short ($/unit)"

    # Verification Checks
    cdf_z = STANDARD_NORMAL.cdf(z_star)
    demand_bal = abs((expected_sales + expected_shortage) - mu)
    supply_bal = abs((expected_sales + expected_leftover) - recommended_qty)
    verification_checks = [
        _make_check(
            "Critical Ratio Formula & Bounds",
            (0.0 < critical_ratio < 1.0) and abs(critical_ratio - (Cu / (Cu + Co))) < 1e-9,
            f"CR = {Cu:,.4f} / ({Cu:,.4f} + {Co:,.4f}) = {critical_ratio:.6f} is strictly within (0, 1).",
        ),
        _make_check(
            "Normal CDF Round-Trip Accuracy",
            abs(cdf_z - critical_ratio) < 1e-6,
            f"Phi({z_star:.4f}) = {cdf_z:.6f} matches CR = {critical_ratio:.6f}.",
        ),
        _make_check(
            "Demand & Supply Conservation Identities",
            (demand_bal < 1e-6) and (supply_bal < 1e-6),
            f"E[Sales]+E[Shortage]={expected_sales + expected_shortage:,.4f} (mu={mu:,.4f}); "
            f"E[Sales]+E[Leftover]={expected_sales + expected_leftover:,.4f} (Q*={recommended_qty:,.4f}).",
        ),
    ]
    if not clamped_at_zero:
        def _mismatch(q_val: float) -> float:
            zz = (q_val - mu) / sigma
            short = sigma * standard_normal_loss(zz)
            sales = mu - short
            left = q_val - sales
            return Cu * short + Co * left

        mc_low = _mismatch(recommended_qty - 0.01 * sigma)
        mc_high = _mismatch(recommended_qty + 0.01 * sigma)
        verification_checks.append(
            _make_check(
                "Local Minimum of Expected Mismatch Cost",
                (mc_low > expected_mismatch_cost) and (mc_high > expected_mismatch_cost),
                f"Mismatch(Q*-0.01*sigma)=${mc_low:,.4f} > Mismatch(Q*)=${expected_mismatch_cost:,.4f} < Mismatch(Q*+0.01*sigma)=${mc_high:,.4f}.",
            )
        )
    if derived_from_price_cost and expected_profit is not None and p is not None and c is not None:
        alt_profit = (p - c) * mu - expected_mismatch_cost
        verification_checks.append(
            _make_check(
                "Expected Profit Cross-Check",
                abs(alt_profit - expected_profit) < 1e-6,
                f"Direct revenue-cost profit (${expected_profit:,.2f}) equals max margin - mismatch cost (${alt_profit:,.2f}).",
            )
        )

    return {
        "model": "newsvendor",
        "inputs_extracted": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formula_with_substituted_values": substituted,
        "intermediate_calculations": {
            "underage_cost": Cu,
            "overage_cost": Co,
            "critical_ratio": critical_ratio,
            "z_score": z_star,
            "unclamped_order_qty": unclamped_q,
            "clamped_at_zero": clamped_at_zero,
            "standard_normal_loss": loss_val,
            "expected_sales": expected_sales,
            "expected_shortage": expected_shortage,
            "expected_leftover": expected_leftover,
            "expected_mismatch_cost": expected_mismatch_cost,
            "fill_rate": fill_rate,
            "in_stock_probability": in_stock_prob,
        },
        "final_answer": {
            "underage_cost": Cu,
            "overage_cost": Co,
            "critical_ratio": critical_ratio,
            "recommended_order_qty": recommended_qty,
            "expected_profit": expected_profit,
        },
        "verification_checks": verification_checks,
        "assumptions": assumptions,
    }


# Backwards-compatible aliases
calculate_safety_stock_rop = calculate_rop
calculate_fixed_period_review = calculate_periodic_review


# ---------------------------------------------------------------------------
# 6. Multicriteria Inventory Classification (A, B, C, D) using Sales ($) & Hits
# ---------------------------------------------------------------------------

CLASSIFICATION_REQUIRED_COLUMNS = [
    "Item Number",
    "Item Name",
    "UoM",
    "Sales ($)",
    "Hits",
]

CLASS_TO_SCORE = {"A": 4, "B": 3, "C": 2, "D": 1}


def _assign_pareto_letter(cum_pct: float, value: float) -> str:
    """Assign A/B/C/D using separate Pareto cumulative thresholds:
    Class A: 0-60%, Class B: 61-80%, Class C: 81-90%, Class D: 91-100%.
    """
    if value <= 0.0:
        return "D"
    if cum_pct <= 60.0 + 1e-9:
        return "A"
    if cum_pct <= 80.0 + 1e-9:
        return "B"
    if cum_pct <= 90.0 + 1e-9:
        return "C"
    return "D"


def _assign_final_class_from_score(combined_score: float) -> str:
    """Assign Final Inventory Class from Combined Weighted Score:
    Class A: 4.00 - 3.25
    Class B: 3.24 - 2.50
    Class C: 2.49 - 1.75
    Class D: Less than 1.75
    """
    if combined_score >= 3.25 - 1e-9:
        return "A"
    if combined_score >= 2.50 - 1e-9:
        return "B"
    if combined_score >= 1.75 - 1e-9:
        return "C"
    return "D"


def run_multicriteria_classification(
    df: Any,
    sales_weight: float = 0.65,
    hits_weight: float = 0.35,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Perform deterministic pandas multicriteria inventory classification (A, B, C, D):

    Step 1: Rank Sales ($) and POS Hits separately using Pareto thresholds:
            Class A: 0-60%, Class B: 61-80%, Class C: 81-90%, Class D: 91-100%.
    Step 2: Convert each class to numerical scores: A = 4, B = 3, C = 2, D = 1.
    Step 3: Calculate Combined Weighted Score:
            Combined Score = 0.65 * (Sales Score) + 0.35 * (Hits Score).
    Step 4: Assign Final Inventory Class from Combined Score:
            Class A: 4.00 - 3.25, Class B: 3.24 - 2.50,
            Class C: 2.49 - 1.75, Class D: < 1.75.

    Returns
    -------
    (classified_df, class_summary_df, report_dict)
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise InventoryValidationError("Input `df` must be a pandas DataFrame.")
    if df.empty:
        raise InventoryValidationError("Uploaded inventory DataFrame is empty; no items to classify.")

    # Support `Hits` or `POS Hits` column naming seamlessly
    work_df = df.copy()
    hits_col = "Hits" if "Hits" in work_df.columns else ("POS Hits" if "POS Hits" in work_df.columns else "Hits")
    required_cols = ["Item Number", "Item Name", "UoM", "Sales ($)", hits_col]
    missing_cols = [col for col in required_cols if col not in work_df.columns]
    if missing_cols:
        raise InventoryValidationError(
            "Missing required column(s) for inventory classification: "
            + ", ".join(missing_cols)
            + "."
        )

    w_sales_raw = _validate_non_negative("sales_weight", sales_weight)
    w_hits_raw = _validate_non_negative("hits_weight", hits_weight)
    if (w_sales_raw + w_hits_raw) <= 0.0:
        raise InventoryValidationError("sales_weight + hits_weight must be strictly positive.")
    w_sales = w_sales_raw / (w_sales_raw + w_hits_raw)
    w_hits = w_hits_raw / (w_sales_raw + w_hits_raw)

    # Clean currency / comma strings via pandas if needed
    if not pd.api.types.is_numeric_dtype(work_df["Sales ($)"]):
        work_df["Sales ($)"] = (
            work_df["Sales ($)"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    if not pd.api.types.is_numeric_dtype(work_df[hits_col]):
        work_df[hits_col] = (
            work_df[hits_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

    work_df["Sales ($)"] = pd.to_numeric(work_df["Sales ($)"], errors="coerce")
    work_df[hits_col] = pd.to_numeric(work_df[hits_col], errors="coerce")

    if work_df["Sales ($)"].isna().any() or work_df[hits_col].isna().any():
        raise InventoryValidationError(
            f"Columns `Sales ($)` and `{hits_col}` must contain valid numeric values for every row."
        )
    if (work_df["Sales ($)"] < 0).any() or (work_df[hits_col] < 0).any():
        raise InventoryValidationError(
            f"Columns `Sales ($)` and `{hits_col}` must be non-negative."
        )

    work_df = work_df.reset_index(drop=True)
    total_items = int(len(work_df))
    total_sales = float(work_df["Sales ($)"].sum())
    total_hits = float(work_df[hits_col].sum())

    # ------------------------------------------------------------------
    # Step 1A: Rank Sales ($) separately using Pareto thresholds
    # ------------------------------------------------------------------
    sales_ranked = work_df[["Sales ($)"]].sort_values(
        by="Sales ($)", ascending=False, kind="mergesort"
    )
    sales_pct = (
        (sales_ranked["Sales ($)"] / total_sales) * 100.0
        if total_sales > 0.0
        else pd.Series(0.0, index=sales_ranked.index)
    )
    cum_sales_pct = sales_pct.cumsum()
    sales_class = pd.Series(
        [
            _assign_pareto_letter(float(c), float(v))
            for c, v in zip(cum_sales_pct, sales_ranked["Sales ($)"])
        ],
        index=sales_ranked.index,
    )

    work_df["Sales %"] = sales_pct.reindex(work_df.index)
    work_df["Cumulative Sales %"] = cum_sales_pct.reindex(work_df.index)
    work_df["Sales Class"] = sales_class.reindex(work_df.index)

    # ------------------------------------------------------------------
    # Step 1B: Rank POS Hits separately using Pareto thresholds
    # ------------------------------------------------------------------
    hits_ranked = work_df[[hits_col]].sort_values(
        by=hits_col, ascending=False, kind="mergesort"
    )
    hits_pct = (
        (hits_ranked[hits_col] / total_hits) * 100.0
        if total_hits > 0.0
        else pd.Series(0.0, index=hits_ranked.index)
    )
    cum_hits_pct = hits_pct.cumsum()
    hits_class = pd.Series(
        [
            _assign_pareto_letter(float(c), float(v))
            for c, v in zip(cum_hits_pct, hits_ranked[hits_col])
        ],
        index=hits_ranked.index,
    )

    work_df["Hits %"] = hits_pct.reindex(work_df.index)
    work_df["Cumulative Hits %"] = cum_hits_pct.reindex(work_df.index)
    work_df["Hits Class"] = hits_class.reindex(work_df.index)

    # ------------------------------------------------------------------
    # Step 2: Convert each class to numerical scores (A=4, B=3, C=2, D=1)
    # ------------------------------------------------------------------
    work_df["Sales Score"] = work_df["Sales Class"].map(CLASS_TO_SCORE).astype(int)
    work_df["Hits Score"] = work_df["Hits Class"].map(CLASS_TO_SCORE).astype(int)

    # ------------------------------------------------------------------
    # Step 3: Combined Weighted Score = 0.65 * Sales Score + 0.35 * Hits Score
    # ------------------------------------------------------------------
    work_df["Combined Score"] = (
        w_sales * work_df["Sales Score"] + w_hits * work_df["Hits Score"]
    ).round(4)

    # ------------------------------------------------------------------
    # Step 4: Assign Final Inventory Class from Combined Score
    # ------------------------------------------------------------------
    work_df["Final Inventory Class"] = work_df["Combined Score"].apply(
        lambda s: _assign_final_class_from_score(float(s))
    )

    # Sort classified dataframe by Combined Score descending, then Sales ($) and Hits descending
    work_df = work_df.sort_values(
        by=["Combined Score", "Sales ($)", hits_col],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    # Build ordered A, B, C, D class summary DataFrame strictly via pandas
    class_order = ["A", "B", "C", "D"]
    grouped = (
        work_df.groupby("Final Inventory Class", as_index=False)
        .agg(
            item_count=("Item Number", "count"),
            total_sales=("Sales ($)", "sum"),
            total_hits=(hits_col, "sum"),
        )
    )
    all_classes_df = pd.DataFrame({"Final Inventory Class": class_order})
    class_summary_df = all_classes_df.merge(
        grouped, on="Final Inventory Class", how="left"
    ).fillna({"item_count": 0, "total_sales": 0.0, "total_hits": 0.0})
    class_summary_df["item_count"] = class_summary_df["item_count"].astype(int)
    class_summary_df["% of Items"] = (class_summary_df["item_count"] / total_items) * 100.0
    class_summary_df["% of Sales"] = (
        (class_summary_df["total_sales"] / total_sales) * 100.0 if total_sales > 0.0 else 0.0
    )
    class_summary_df["% of Hits"] = (
        (class_summary_df["total_hits"] / total_hits) * 100.0 if total_hits > 0.0 else 0.0
    )
    class_summary_df = class_summary_df.rename(
        columns={
            "item_count": "Item Count",
            "total_sales": "Total Sales ($)",
            "total_hits": "Total Hits",
        }
    )

    class_counts = {
        row["Final Inventory Class"]: int(row["Item Count"])
        for _, row in class_summary_df.iterrows()
    }
    class_total_sales = {
        row["Final Inventory Class"]: float(row["Total Sales ($)"])
        for _, row in class_summary_df.iterrows()
    }

    # Verification Checks computed deterministically via pandas
    valid_letters = {"A", "B", "C", "D"}
    classes_valid = (
        set(work_df["Sales Class"].unique()).issubset(valid_letters)
        and set(work_df["Hits Class"].unique()).issubset(valid_letters)
        and set(work_df["Final Inventory Class"].unique()).issubset(valid_letters)
    )
    sales_pct_sum = float(work_df["Sales %"].sum())
    hits_pct_sum = float(work_df["Hits %"].sum())
    scores_valid = bool(
        (work_df["Sales Score"] == work_df["Sales Class"].map(CLASS_TO_SCORE)).all()
        and (work_df["Hits Score"] == work_df["Hits Class"].map(CLASS_TO_SCORE)).all()
    )
    recomputed_combined = (w_sales * work_df["Sales Score"] + w_hits * work_df["Hits Score"]).round(4)
    combined_ok = bool(((work_df["Combined Score"] - recomputed_combined).abs() < 1e-6).all())
    final_class_ok = bool(
        (
            work_df["Final Inventory Class"]
            == work_df["Combined Score"].apply(lambda s: _assign_final_class_from_score(float(s)))
        ).all()
    )
    summary_item_sum = int(class_summary_df["Item Count"].sum())
    summary_sales_sum = float(class_summary_df["Total Sales ($)"].sum())
    summary_hits_sum = float(class_summary_df["Total Hits"].sum())

    verification_checks = [
        _make_check(
            "Complete & Valid A/B/C/D Assignments",
            classes_valid and summary_item_sum == total_items,
            f"All {summary_item_sum:,}/{total_items:,} items have valid Sales, Hits, and Final Inventory Classes in {{A, B, C, D}}.",
        ),
        _make_check(
            "Separate Pareto Shares Sum to 100%",
            (total_sales == 0.0 or abs(sales_pct_sum - 100.0) < 1e-6)
            and (total_hits == 0.0 or abs(hits_pct_sum - 100.0) < 1e-6),
            f"Sales % sum = {sales_pct_sum:.4f}%; Hits % sum = {hits_pct_sum:.4f}%.",
        ),
        _make_check(
            "Class-to-Score Mapping (A=4, B=3, C=2, D=1)",
            scores_valid,
            "Every row's Sales Score and Hits Score matches the A=4, B=3, C=2, D=1 mapping.",
        ),
        _make_check(
            "Combined Weighted Score & Final Class Thresholds",
            combined_ok and final_class_ok,
            f"Every row satisfies Combined Score = {w_sales:.2f}(Sales Score) + {w_hits:.2f}(Hits Score) "
            "and Final Class thresholds (A: 4.00-3.25, B: 3.24-2.50, C: 2.49-1.75, D: < 1.75).",
        ),
        _make_check(
            "Class Totals Reconcile with Dataset Totals",
            abs(summary_sales_sum - total_sales) < 1e-6 and abs(summary_hits_sum - total_hits) < 1e-6,
            f"Class Sales sum = ${summary_sales_sum:,.2f} (total ${total_sales:,.2f}); "
            f"Class Hits sum = {_fmt_num(summary_hits_sum)} (total {_fmt_num(total_hits)}).",
        ),
    ]

    variable_definitions: Dict[str, str] = {
        "Sales ($)": "Monetary sales volume per inventory item ($)",
        "Hits": "POS pick / order frequency per inventory item (count)",
        "Sales Class": "Separate Pareto class from Cumulative Sales % (A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%)",
        "Hits Class": "Separate Pareto class from Cumulative Hits % (A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%)",
        "Sales Score": "Numerical score converted from Sales Class (A=4, B=3, C=2, D=1)",
        "Hits Score": "Numerical score converted from Hits Class (A=4, B=3, C=2, D=1)",
        "Combined Score": f"Weighted multicriteria score = {w_sales:.2f}(Sales Score) + {w_hits:.2f}(Hits Score)",
        "Final Inventory Class": "Final class from Combined Score (A: 4.00-3.25, B: 3.24-2.50, C: 2.49-1.75, D: <1.75)",
    }

    formulas = [
        "Step 1 (Separate Pareto Ranking): Sort Sales ($) descending -> Cumulative Sales %; Sort Hits descending -> Cumulative Hits %",
        "Step 1 Thresholds (both Sales and Hits): Class A = 0-60%, Class B = 61-80%, Class C = 81-90%, Class D = 91-100%",
        "Step 2 (Class to Score): A = 4, B = 3, C = 2, D = 1",
        f"Step 3 (Combined Weighted Score): Combined Score = {w_sales:.2f} * (Sales Score) + {w_hits:.2f} * (Hits Score)",
        "Step 4 (Final Inventory Class): A = 4.00-3.25, B = 3.24-2.50, C = 2.49-1.75, D = Less than 1.75",
    ]

    class_summary_line = "; ".join(
        f"Class {cls}: {class_counts[cls]:,} items (${class_total_sales[cls]:,.2f})"
        for cls in class_order
    )
    substituted = [
        f"Total Items = {total_items:,}; Total Sales ($) = ${total_sales:,.2f}; Total Hits = {_fmt_num(total_hits)}",
        f"Combined Score_i = {w_sales:.2f} * (Sales Score_i) + {w_hits:.2f} * (Hits Score_i)",
        f"Final Class Summary -> {class_summary_line}",
    ]

    assumptions = [
        "Sales ($) and POS Hits are ranked independently in descending order to compute separate Pareto cumulative percentages.",
        "Separate Pareto cutoffs: Class A (<= 60%), Class B (> 60% to <= 80%), Class C (> 80% to <= 90%), Class D (> 90% or zero activity).",
        "Discrete class scores map A = 4, B = 3, C = 2, D = 1.",
        f"Combined Score weights are {w_sales * 100.0:.0f}% Sales Score and {w_hits * 100.0:.0f}% Hits Score.",
        "Final Inventory Class intervals: A (4.00-3.25), B (3.24-2.50), C (2.49-1.75), D (< 1.75).",
    ]

    report: Dict[str, Any] = {
        "model": "inventory_classification",
        "inputs_extracted": {
            "row_count": total_items,
            "columns": list(df.columns),
            "sales_weight": w_sales,
            "hits_weight": w_hits,
            "pareto_thresholds": {
                "A": "0-60%",
                "B": "61-80%",
                "C": "81-90%",
                "D": "91-100%",
            },
            "score_mapping": dict(CLASS_TO_SCORE),
            "final_class_thresholds": {
                "A": "4.00 - 3.25",
                "B": "3.24 - 2.50",
                "C": "2.49 - 1.75",
                "D": "< 1.75",
            },
        },
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formula_with_substituted_values": substituted,
        "intermediate_calculations": {
            "total_items": total_items,
            "total_sales": total_sales,
            "total_hits": total_hits,
            "class_summary": class_summary_df.to_dict(orient="records"),
        },
        "final_answer": {
            "class_counts": class_counts,
            "class_total_sales": class_total_sales,
            "total_items_classified": total_items,
            "total_sales": total_sales,
            "total_hits": total_hits,
        },
        "verification_checks": verification_checks,
        "assumptions": assumptions,
    }

    return work_df, class_summary_df, report


def classify_inventory_multicriteria(
    df: Any,
    sales_weight: float = 0.65,
    hits_weight: float = 0.35,
) -> Dict[str, Any]:
    """Return the JSON-serializable 8-part multicriteria classification report."""
    _, _, report = run_multicriteria_classification(
        df=df,
        sales_weight=sales_weight,
        hits_weight=hits_weight,
    )
    return report


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    model_id: str
    display_name: str
    function_name: str
    func: Callable[..., Dict[str, Any]]
    required_params: List[str]
    optional_params: List[str] = field(default_factory=list)
    alternative_param_groups: List[List[str]] = field(default_factory=list)
    description: str = ""


TOOL_REGISTRY: Dict[str, ToolSpec] = {
    "eoq": ToolSpec(
        model_id="eoq",
        display_name="Economic Order Quantity (EOQ)",
        function_name="calculate_eoq",
        func=calculate_eoq,
        required_params=["annual_demand", "ordering_cost"],
        optional_params=[
            "holding_cost",
            "unit_cost",
            "holding_cost_rate",
            "working_days_per_year",
            "lead_time_days",
        ],
        alternative_param_groups=[
            ["holding_cost"],
            ["unit_cost", "holding_cost_rate"],
        ],
        description="Optimal fixed order quantity Q = sqrt(2DS/H) minimizing annual ordering and holding costs.",
    ),
    "quantity_discount": ToolSpec(
        model_id="quantity_discount",
        display_name="Quantity Discount (All-Units)",
        function_name="calculate_quantity_discount",
        func=calculate_quantity_discount,
        required_params=["annual_demand", "ordering_cost", "price_breaks"],
        optional_params=["holding_cost_rate", "holding_cost"],
        alternative_param_groups=[
            ["holding_cost_rate"],
            ["holding_cost"],
        ],
        description="Evaluates feasible quantities across all-units price breaks under fixed or percentage holding cost.",
    ),
    "safety_stock_rop": ToolSpec(
        model_id="safety_stock_rop",
        display_name="Safety Stock & Reorder Point",
        function_name="calculate_rop",
        func=calculate_rop,
        required_params=["avg_demand", "lead_time"],
        optional_params=[
            "demand_std",
            "lead_time_std",
            "service_level",
            "z_score",
            "demand_period",
        ],
        description="Computes safety stock and reorder point across all 4 demand/lead-time variability regimes.",
    ),
    "fixed_period_review": ToolSpec(
        model_id="fixed_period_review",
        display_name="Fixed-Period Review (Order-Up-To)",
        function_name="calculate_periodic_review",
        func=calculate_periodic_review,
        required_params=["avg_demand", "review_period", "lead_time"],
        optional_params=[
            "demand_std",
            "service_level",
            "z_score",
            "safety_stock",
            "inventory_position",
            "on_hand_inventory",
            "on_order_inventory",
            "backorders",
            "lead_time_std",
            "demand_period",
        ],
        description="Computes protection period, target inventory, safety stock, and order quantity for periodic review.",
    ),
    "newsvendor": ToolSpec(
        model_id="newsvendor",
        display_name="Newsvendor (Single-Period Model)",
        function_name="calculate_newsvendor",
        func=calculate_newsvendor,
        required_params=["mean_demand", "demand_std"],
        optional_params=[
            "selling_price",
            "unit_cost",
            "salvage_value",
            "stockout_penalty",
            "underage_cost",
            "overage_cost",
            "demand_distribution",
        ],
        alternative_param_groups=[
            ["underage_cost", "overage_cost"],
            ["selling_price", "unit_cost", "salvage_value"],
        ],
        description="Calculates underage cost, overage cost, critical ratio, and recommended single-period order quantity.",
    ),
    "inventory_classification": ToolSpec(
        model_id="inventory_classification",
        display_name="Multicriteria Inventory Classification (A/B/C/D)",
        function_name="classify_inventory_multicriteria",
        func=classify_inventory_multicriteria,
        required_params=["df"],
        optional_params=[
            "sales_weight",
            "hits_weight",
        ],
        description="Separate Pareto ranking of Sales ($) and Hits (0-60/61-80/81-90/91-100%), scored A=4..D=1, weighted 65% Sales / 35% Hits into Final A/B/C/D classes.",
    ),
}
