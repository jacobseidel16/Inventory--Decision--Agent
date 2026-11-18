"""Streamlit Web Application (and optional CLI) for the Supply Chain Analytics
Inventory Decision Agent.

Run the interactive dashboard:
    streamlit run app.py

Or run directly in CLI mode:
    python app.py --cli "Annual demand is 12,000 units, ordering cost is $50, holding cost is $4. What is EOQ?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from inventory_agent import AgentResponse, InventoryDecisionAgent
from inventory_tools import (
    TOOL_REGISTRY,
    InventoryValidationError,
    run_multicriteria_classification,
)

EXAMPLE_QUESTIONS: Dict[str, str] = {
    "1. Economic Order Quantity (EOQ)": (
        "Our warehouse has an annual demand of 12,000 units with 300 working days per year. "
        "Ordering cost is $75 per order, unit cost is $25, and the annual holding cost rate is 20%. "
        "Lead time is 5 days. What is our optimal EOQ and total annual inventory cost?"
    ),
    "2. Quantity Discount": (
        "Annual demand is 5,000 units, ordering cost is $49 per order, and carrying cost is 20% "
        "of unit price. The supplier offers all-units quantity discounts: 1-999 units at $5.00, "
        "1,000-1,999 units at $4.80, and 2,000+ units at $4.75. Which order quantity minimizes total annual cost?"
    ),
    "3. Safety Stock & Reorder Point": (
        "Daily demand averages 100 units with a standard deviation of 20 units. Supplier lead time "
        "averages 5 days with a lead time standard deviation of 1 day. We want a 95% cycle service level. "
        "What should our safety stock and reorder point be?"
    ),
    "4. Fixed-Period Review": (
        "We use a fixed-period review policy and review inventory every 14 days. Average daily demand "
        "is 50 units with a standard deviation of 10 units, lead time is 3 days, and target service level "
        "is 98%. We currently have 400 units on hand and 100 units on order. What is the target inventory "
        "level and how many units should we order now?"
    ),
    "5. Newsvendor (Single-Period)": (
        "We are ordering seasonal winter jackets for a single selling season. Mean demand is 500 units "
        "with a standard deviation of 100 units. We sell each jacket for $120, purchase cost is $70 per unit, "
        "and unsold jackets have a salvage value of $30. How many jackets should we order?"
    ),
}


def run_cli(question: str | None = None, use_llm: bool = False) -> None:
    """Run the Inventory Decision Agent from the command line."""
    agent = InventoryDecisionAgent(use_llm=use_llm)

    def _display(resp: AgentResponse) -> None:
        print("=" * 72)
        print(f"STATUS: {resp.status.upper()}")
        print(
            f"[1] Identify Model   : {resp.identification.display_name} "
            f"(confidence={resp.identification.confidence:.0%})"
        )
        print(f"[2] Extract Inputs   : {json.dumps(resp.extraction.inputs)}")
        if resp.tool_selection:
            print(f"[3] Select Tool      : {resp.tool_selection.tool_name}")
        if resp.verification:
            passed = sum(1 for c in resp.verification.checks if c.passed)
            print(
                f"[5] Verify           : {'PASSED' if resp.verification.passed else 'FAILED'} "
                f"({passed}/{len(resp.verification.checks)} checks)"
            )
        print("-" * 72)
        print(resp.explanation)
        print("=" * 72)

    if question:
        _display(agent.answer(question))
        return

    print("Supply Chain Analytics Inventory Decision Agent (CLI Mode)")
    print("Type a natural-language inventory question, or 'quit' to exit.\n")
    while True:
        try:
            q = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not q or q.lower() in {"quit", "exit", "q"}:
            break
        _display(agent.answer(q))


def _render_streamlit_app() -> None:
    import pandas as pd
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
    )

    st.title("Supply Chain Analytics Inventory Decision Agent")
    st.caption(
        "Architecture: **Customer Question → Identify Model → Extract Inputs → "
        "Select Python Tool → Calculate → Verify → Explain**"
    )

    with st.sidebar:
        st.header("Supported Python Tools")
        for spec in TOOL_REGISTRY.values():
            st.markdown(f"- **{spec.display_name}** (`{spec.function_name}()`)")

        st.divider()
        st.header("Agent Settings")
        has_api_key = bool(os.getenv("OPENAI_API_KEY"))
        use_llm = st.checkbox(
            "Enable optional OpenAI fallback extraction",
            value=False,
            disabled=not has_api_key,
            help=(
                "Uses OpenAI only as a fallback parameter extractor when OPENAI_API_KEY is set. "
                "All calculations always run strictly in Python."
            ),
        )
        if not has_api_key:
            st.caption("Running in 100% offline deterministic Python mode (no API key needed).")

        st.divider()
        st.header("Load Example Question")
        for label, prompt in EXAMPLE_QUESTIONS.items():
            if st.button(label, use_container_width=True):
                st.session_state["nl_question"] = prompt
                st.session_state["auto_run"] = True

        st.divider()
        st.header("Data Upload")
        uploaded_csv = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            accept_multiple_files=False,
        )

    # Read the uploaded CSV once into a pandas DataFrame shared across the app
    uploaded_df: Any = None
    upload_read_error: str | None = None
    if uploaded_csv is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_csv)
            st.session_state["uploaded_df"] = uploaded_df
        except Exception as exc:
            upload_read_error = str(exc)
            st.session_state["uploaded_df"] = None
    else:
        st.session_state["uploaded_df"] = None

    agent = InventoryDecisionAgent(use_llm=use_llm)

    tab_ask, tab_calc = st.tabs(["Natural-Language Agent", "Direct Python Calculator"])

    with tab_ask:
        default_q = st.session_state.get("nl_question", EXAMPLE_QUESTIONS["1. Economic Order Quantity (EOQ)"])
        question = st.text_area(
            "Ask an inventory optimization question in natural language:",
            value=default_q,
            height=120,
            key="question_input",
        )
        col_btn, _ = st.columns([1, 5])
        with col_btn:
            run_clicked = st.button("Analyze & Decide", type="primary", use_container_width=True)

        should_run = run_clicked or st.session_state.pop("auto_run", False)
        if should_run:
            response = agent.answer(question, uploaded_df=uploaded_df)
            st.session_state["last_response"] = response

        response: AgentResponse | None = st.session_state.get("last_response")
        if response is not None:
            _render_response(st, pd, response)

    with tab_calc:
        st.subheader("Structured Input Mode")
        numeric_models = [k for k in TOOL_REGISTRY.keys() if k != "inventory_classification"]
        model_choice = st.selectbox(
            "Select Inventory Model",
            options=numeric_models,
            format_func=lambda k: f"{TOOL_REGISTRY[k].display_name} — {TOOL_REGISTRY[k].function_name}()",
        )

        override_inputs: Dict[str, Any] = {}
        if model_choice == "eoq":
            c1, c2, c3 = st.columns(3)
            override_inputs["annual_demand"] = c1.number_input("Annual Demand (D)", min_value=1.0, value=12000.0)
            override_inputs["ordering_cost"] = c2.number_input("Ordering Cost (S, $)", min_value=0.01, value=75.0)
            hc_mode = c3.radio("Holding Cost Input", ["Direct ($/unit/yr)", "Rate (% of Unit Cost)"], horizontal=True)
            c4, c5, c6 = st.columns(3)
            if hc_mode == "Direct ($/unit/yr)":
                override_inputs["holding_cost"] = c4.number_input("Holding Cost (H, $/unit/yr)", min_value=0.01, value=5.0)
                unit_c = c5.number_input("Optional Unit Cost ($, 0 to omit)", min_value=0.0, value=0.0)
                if unit_c > 0:
                    override_inputs["unit_cost"] = unit_c
            else:
                override_inputs["unit_cost"] = c4.number_input("Unit Cost (C, $)", min_value=0.01, value=25.0)
                rate_pct = c5.number_input("Annual Holding Rate (%)", min_value=0.1, max_value=200.0, value=20.0)
                override_inputs["holding_cost_rate"] = rate_pct / 100.0
            wd = c6.number_input("Working Days / Year (0 to omit)", min_value=0.0, max_value=366.0, value=300.0)
            if wd > 0:
                override_inputs["working_days_per_year"] = wd

        elif model_choice == "quantity_discount":
            c1, c2, c3 = st.columns(3)
            override_inputs["annual_demand"] = c1.number_input("Annual Demand (D)", min_value=1.0, value=5000.0, key="qd_d")
            override_inputs["ordering_cost"] = c2.number_input("Ordering Cost (S, $)", min_value=0.01, value=49.0, key="qd_s")
            qd_hc_mode = c3.radio("Holding Cost Type", ["Percentage (% of price)", "Fixed ($/unit/yr)"], horizontal=True)
            if qd_hc_mode == "Percentage (% of price)":
                rate_pct = st.number_input("Holding Cost Rate (%)", min_value=0.1, max_value=200.0, value=20.0, key="qd_h")
                override_inputs["holding_cost_rate"] = rate_pct / 100.0
            else:
                override_inputs["holding_cost"] = st.number_input("Fixed Holding Cost ($/unit/yr)", min_value=0.01, value=2.0, key="qd_hf")

            st.caption("Edit all-units price break tiers (leave `max_qty` blank for the highest tier):")
            default_tiers = pd.DataFrame(
                [
                    {"min_qty": 1.0, "max_qty": 999.0, "unit_cost": 5.00},
                    {"min_qty": 1000.0, "max_qty": 1999.0, "unit_cost": 4.80},
                    {"min_qty": 2000.0, "max_qty": None, "unit_cost": 4.75},
                ]
            )
            edited_df = st.data_editor(default_tiers, num_rows="dynamic", use_container_width=True)
            tiers_list: List[Dict[str, Any]] = []
            for _, row in edited_df.iterrows():
                if pd.notna(row["min_qty"]) and pd.notna(row["unit_cost"]):
                    tiers_list.append(
                        {
                            "min_qty": float(row["min_qty"]),
                            "max_qty": float(row["max_qty"]) if pd.notna(row["max_qty"]) else None,
                            "unit_cost": float(row["unit_cost"]),
                        }
                    )
            override_inputs["price_breaks"] = tiers_list

        elif model_choice == "safety_stock_rop":
            c1, c2, c3 = st.columns(3)
            override_inputs["avg_demand"] = c1.number_input("Average Demand per Period (d)", min_value=0.01, value=100.0)
            override_inputs["demand_std"] = c2.number_input("Demand Std. Dev. (sigma_d, 0 = constant)", min_value=0.0, value=20.0)
            override_inputs["lead_time"] = c3.number_input("Average Lead Time (L)", min_value=0.01, value=5.0)
            c4, c5, c6 = st.columns(3)
            override_inputs["lead_time_std"] = c4.number_input("Lead Time Std. Dev. (sigma_L, 0 = constant)", min_value=0.0, value=1.0)
            if override_inputs["demand_std"] > 0 or override_inputs["lead_time_std"] > 0:
                csl_pct = c5.number_input("Cycle Service Level (%)", min_value=50.0, max_value=99.99, value=95.0)
                override_inputs["service_level"] = csl_pct / 100.0
            override_inputs["demand_period"] = c6.selectbox("Time Unit", ["day", "week", "month"])

        elif model_choice == "fixed_period_review":
            c1, c2, c3 = st.columns(3)
            override_inputs["avg_demand"] = c1.number_input("Average Demand per Period (d)", min_value=0.01, value=50.0, key="fp_d")
            override_inputs["demand_std"] = c2.number_input("Demand Std. Dev. (sigma_d)", min_value=0.0, value=10.0, key="fp_sd")
            override_inputs["review_period"] = c3.number_input("Review Period (T)", min_value=0.01, value=14.0, key="fp_t")
            c4, c5, c6 = st.columns(3)
            override_inputs["lead_time"] = c4.number_input("Lead Time (L)", min_value=0.0, value=3.0, key="fp_l")
            if override_inputs["demand_std"] > 0:
                csl_pct = c5.number_input("Cycle Service Level (%)", min_value=50.0, max_value=99.99, value=98.0, key="fp_sl")
                override_inputs["service_level"] = csl_pct / 100.0
            override_inputs["demand_period"] = c6.selectbox("Time Unit", ["day", "week", "month"], key="fp_u")
            c7, c8, c9 = st.columns(3)
            override_inputs["on_hand_inventory"] = c7.number_input("On-Hand Inventory", min_value=0.0, value=400.0)
            override_inputs["on_order_inventory"] = c8.number_input("On-Order Inventory", min_value=0.0, value=100.0)
            override_inputs["backorders"] = c9.number_input("Backorders", min_value=0.0, value=0.0)

        elif model_choice == "newsvendor":
            c1, c2, c3 = st.columns(3)
            override_inputs["mean_demand"] = c1.number_input("Mean Demand (mu)", min_value=0.01, value=500.0)
            override_inputs["demand_std"] = c2.number_input("Demand Std. Dev. (sigma)", min_value=0.01, value=100.0)
            nv_mode = c3.radio("Cost Specification", ["Price / Cost / Salvage", "Direct Cu & Co"], horizontal=True)
            c4, c5, c6 = st.columns(3)
            if nv_mode == "Price / Cost / Salvage":
                override_inputs["selling_price"] = c4.number_input("Selling Price (p, $)", min_value=0.01, value=120.0)
                override_inputs["unit_cost"] = c5.number_input("Unit Cost (c, $)", min_value=0.01, value=70.0)
                override_inputs["salvage_value"] = c6.number_input("Salvage Value (v, $; negative = disposal)", value=30.0)
                pen = st.number_input("Optional Goodwill/Stockout Penalty (g, $, 0 to omit)", min_value=0.0, value=0.0)
                if pen > 0:
                    override_inputs["stockout_penalty"] = pen
            else:
                override_inputs["underage_cost"] = c4.number_input("Underage Cost (Cu, $)", min_value=0.01, value=50.0)
                override_inputs["overage_cost"] = c5.number_input("Overage Cost (Co, $)", min_value=0.01, value=40.0)

        if st.button("Calculate & Verify", type="primary", key="calc_direct"):
            direct_resp = agent.answer(
                question=f"Direct calculation for {TOOL_REGISTRY[model_choice].display_name}",
                override_model=model_choice,
                override_inputs=override_inputs,
            )
            _render_response(st, pd, direct_resp)

    if uploaded_csv is not None:
        _render_csv_upload_analysis(st, pd, uploaded_df, upload_read_error)


def _render_csv_upload_analysis(
    st: Any,
    pd: Any,
    uploaded_df: Any,
    upload_read_error: str | None = None,
) -> None:
    """Validate, display, and analyze the uploaded CSV pandas DataFrame.

    Supports both:
    1. Existing Inventory Valuation Analysis (`Product`, `Category`, `Units`, `Unit Cost`)
    2. Multicriteria Inventory Classification (`Item Number`, `Item Name`, `UoM`, `Sales ($)`, `Hits`)
    """
    st.divider()
    st.header("Uploaded CSV Data & Analysis")

    if upload_read_error is not None or uploaded_df is None:
        st.error(f"Unable to read the uploaded CSV file: {upload_read_error or 'Empty or invalid file.'}")
        return

    valuation_required = ["Product", "Category", "Units", "Unit Cost"]
    hits_col = "Hits" if "Hits" in uploaded_df.columns else ("POS Hits" if "POS Hits" in uploaded_df.columns else "Hits")
    classification_required = ["Item Number", "Item Name", "UoM", "Sales ($)", hits_col]

    missing_valuation = [col for col in valuation_required if col not in uploaded_df.columns]
    missing_classification = [col for col in classification_required if col not in uploaded_df.columns]

    has_valuation_schema = len(missing_valuation) == 0
    has_classification_schema = len(missing_classification) == 0

    if not has_valuation_schema and not has_classification_schema:
        st.error(
            "Uploaded CSV does not match a supported analysis schema.\n\n"
            f"- **For Inventory Valuation Analysis**, missing column(s): "
            f"{', '.join(f'`{c}`' for c in missing_valuation)} "
            f"(required: `Product`, `Category`, `Units`, `Unit Cost`).\n"
            f"- **For Multicriteria Inventory Classification (A/B/C/D)**, missing column(s): "
            f"{', '.join(f'`{c}`' for c in missing_classification)} "
            f"(required: `Item Number`, `Item Name`, `UoM`, `Sales ($)`, `Hits`)."
        )
        return

    st.subheader("Uploaded Dataset")
    meta_c1, meta_c2, meta_c3 = st.columns([1, 1, 3])
    meta_c1.metric("Number of Rows", f"{uploaded_df.shape[0]:,}")
    meta_c2.metric("Number of Columns", f"{uploaded_df.shape[1]:,}")
    meta_c3.markdown(
        "**Column Names:** " + ", ".join(f"`{col}`" for col in uploaded_df.columns.tolist())
    )
    st.dataframe(uploaded_df, use_container_width=True)

    if uploaded_df.empty:
        st.warning("The uploaded CSV contains no data rows to analyze.")
        return

    # 1. Existing Inventory Valuation Analysis (Product, Category, Units, Unit Cost)
    if has_valuation_schema:
        _render_inventory_valuation_analysis(st, pd, uploaded_df)

    # 2. Multicriteria Inventory Classification (Item Number, Item Name, UoM, Sales ($), Hits)
    if has_classification_schema:
        _render_inventory_classification_analysis(st, pd, uploaded_df)


def _render_inventory_valuation_analysis(st: Any, pd: Any, uploaded_df: Any) -> None:
    """Run and display existing pandas inventory valuation analysis on the uploaded DataFrame."""
    analysis_df = uploaded_df.copy()
    if not pd.api.types.is_numeric_dtype(analysis_df["Units"]):
        analysis_df["Units"] = (
            analysis_df["Units"].astype(str).str.replace(",", "", regex=False).str.strip()
        )
    if not pd.api.types.is_numeric_dtype(analysis_df["Unit Cost"]):
        analysis_df["Unit Cost"] = (
            analysis_df["Unit Cost"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

    analysis_df["Units"] = pd.to_numeric(analysis_df["Units"], errors="coerce")
    analysis_df["Unit Cost"] = pd.to_numeric(analysis_df["Unit Cost"], errors="coerce")

    if analysis_df["Units"].isna().any() or analysis_df["Unit Cost"].isna().any():
        st.error("Columns `Units` and `Unit Cost` must contain valid numeric values for every row.")
        return

    total_units = analysis_df["Units"].sum()
    avg_unit_cost = analysis_df["Unit Cost"].mean()

    max_units = analysis_df["Units"].max()
    highest_units_products = (
        analysis_df.loc[analysis_df["Units"] == max_units, "Product"].astype(str).tolist()
    )
    highest_units_label = ", ".join(highest_units_products)

    units_by_category = (
        analysis_df.groupby("Category", as_index=False)["Units"]
        .sum()
        .rename(columns={"Units": "Total Units"})
        .sort_values(by="Total Units", ascending=False)
    )

    analysis_df["Inventory Value"] = analysis_df["Units"] * analysis_df["Unit Cost"]
    total_inventory_value = analysis_df["Inventory Value"].sum()

    def _fmt_units(val: float) -> str:
        return f"{int(round(val)):,}" if abs(val - round(val)) < 1e-9 else f"{val:,.2f}"

    st.subheader("Inventory Analysis Results")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total Units", _fmt_units(float(total_units)))
    r2.metric("Average Unit Cost", f"${float(avg_unit_cost):,.2f}")
    r3.metric(
        "Product with Highest Units",
        highest_units_label,
        delta=f"{_fmt_units(float(max_units))} units",
        delta_color="off",
    )
    r4.metric("Total Inventory Value", f"${float(total_inventory_value):,.2f}")

    st.subheader("Total Units by Category")
    st.dataframe(units_by_category, use_container_width=True, hide_index=True)

    st.subheader("Inventory Data with Inventory Value")
    st.dataframe(
        analysis_df,
        use_container_width=True,
        column_config={
            "Unit Cost": st.column_config.NumberColumn("Unit Cost", format="$%.2f"),
            "Inventory Value": st.column_config.NumberColumn("Inventory Value", format="$%.2f"),
        },
    )


def _render_inventory_classification_analysis(st: Any, pd: Any, uploaded_df: Any) -> None:
    """Run and display course-methodology pandas multicriteria A/B/C/D inventory classification."""
    st.subheader("Multicriteria Inventory Classification (A, B, C, D)")
    st.caption(
        "Methodology: Separate Pareto ranking of **Sales ($)** and **POS Hits** "
        "(A: 0–60%, B: 61–80%, C: 81–90%, D: 91–100%) → Score mapping (A=4, B=3, C=2, D=1) → "
        "Combined Score = `0.65(Sales Score) + 0.35(Hits Score)` → "
        "Final Class (A: 4.00–3.25, B: 3.24–2.50, C: 2.49–1.75, D: < 1.75)."
    )

    try:
        classified_df, class_summary_df, report = run_multicriteria_classification(
            df=uploaded_df,
            sales_weight=0.65,
            hits_weight=0.35,
        )
    except InventoryValidationError as exc:
        st.error(f"Inventory classification validation error: {exc}")
        return

    final = report["final_answer"]
    counts = final["class_counts"]
    class_sales = final["class_total_sales"]

    # Final Summary Counts & Total Sales for Each Class (A, B, C, D)
    st.markdown("**Final Summary Counts & Total Sales by Class**")
    c_a, c_b, c_c, c_d = st.columns(4)
    c_a.metric("Class A", f"{counts.get('A', 0):,} items", delta=f"${class_sales.get('A', 0.0):,.2f} sales", delta_color="off")
    c_b.metric("Class B", f"{counts.get('B', 0):,} items", delta=f"${class_sales.get('B', 0.0):,.2f} sales", delta_color="off")
    c_c.metric("Class C", f"{counts.get('C', 0):,} items", delta=f"${class_sales.get('C', 0.0):,.2f} sales", delta_color="off")
    c_d.metric("Class D", f"{counts.get('D', 0):,} items", delta=f"${class_sales.get('D', 0.0):,.2f} sales", delta_color="off")

    t1, t2, t3 = st.columns(3)
    t1.metric("Total Items Classified", f"{final['total_items_classified']:,}")
    t2.metric("Total Sales ($)", f"${final['total_sales']:,.2f}")
    t3.metric("Total Hits", f"{final['total_hits']:,.0f}")

    # Summary Table
    st.markdown("**Final Class Summary Table (Counts, Total Sales & Total Hits)**")
    st.dataframe(
        class_summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Total Sales ($)": st.column_config.NumberColumn("Total Sales ($)", format="$%.2f"),
            "% of Items": st.column_config.NumberColumn("% of Items", format="%.2f%%"),
            "% of Sales": st.column_config.NumberColumn("% of Sales", format="%.2f%%"),
            "% of Hits": st.column_config.NumberColumn("% of Hits", format="%.2f%%"),
        },
    )

    # Charts for Item Count & Total Sales ($) by Final Class
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**Item Count by Final Inventory Class**")
        st.bar_chart(class_summary_df.set_index("Final Inventory Class")[["Item Count"]])
    with ch2:
        st.markdown("**Total Sales ($) by Final Inventory Class**")
        st.bar_chart(class_summary_df.set_index("Final Inventory Class")[["Total Sales ($)"]])

    # Full Classified DataFrame with all step-by-step columns
    st.markdown("**Classified Inventory DataFrame (Step-by-Step Scores & Final Inventory Class)**")
    st.dataframe(
        classified_df,
        use_container_width=True,
        column_config={
            "Sales ($)": st.column_config.NumberColumn("Sales ($)", format="$%.2f"),
            "Sales %": st.column_config.NumberColumn("Sales %", format="%.2f%%"),
            "Cumulative Sales %": st.column_config.NumberColumn("Cumulative Sales %", format="%.2f%%"),
            "Hits %": st.column_config.NumberColumn("Hits %", format="%.2f%%"),
            "Cumulative Hits %": st.column_config.NumberColumn("Cumulative Hits %", format="%.2f%%"),
            "Combined Score": st.column_config.NumberColumn("Combined Score", format="%.2f"),
        },
    )

    csv_bytes = classified_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Classified Inventory CSV",
        data=csv_bytes,
        file_name="classified_inventory_abcd.csv",
        mime="text/csv",
    )

    with st.expander("View Classification Methodology, Verification Checks & Assumptions", expanded=False):
        st.markdown("**Formulas**")
        for expr in report["formula"]:
            st.markdown(f"- `{expr}`")
        st.markdown("**Substituted Summary**")
        for step in report["formula_with_substituted_values"]:
            st.markdown(f"- `{step}`")
        st.markdown("**Verification Checks**")
        chk_rows = [
            {
                "Status": "✅ Pass" if c["passed"] else "❌ Fail",
                "Check": c["name"],
                "Details": c["details"],
            }
            for c in report["verification_checks"]
        ]
        st.dataframe(pd.DataFrame(chk_rows), use_container_width=True, hide_index=True)
        st.markdown("**Assumptions**")
        for a in report["assumptions"]:
            st.markdown(f"- {a}")


def _json_safe(obj: Any) -> Any:
    """Convert DataFrames into readable summaries so st.json/json.dumps never fails."""
    if hasattr(obj, "columns") and hasattr(obj, "shape"):
        return f"<DataFrame ({obj.shape[0]} rows x {obj.shape[1]} columns)>"
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _render_response(st: Any, pd: Any, response: AgentResponse) -> None:
    """Render the full 7-step pipeline trace, visualizations, and explanation."""
    st.divider()

    stages = [
        ("1. Question", True),
        ("2. Identify", response.identification.model_id is not None),
        ("3. Extract", len(response.extraction.missing_fields) == 0 and response.identification.model_id is not None),
        ("4. Select Tool", response.tool_selection is not None),
        ("5. Calculate", response.calculation is not None),
        ("6. Verify", bool(response.verification and response.verification.passed)),
        ("7. Explain", bool(response.explanation)),
    ]
    cols = st.columns(len(stages))
    for col, (label, ok) in zip(cols, stages):
        icon = "✅" if ok else "⚠️"
        col.markdown(f"**{icon} {label}**")

    if response.status == "needs_clarification":
        st.warning(response.explanation)
    elif response.status == "error":
        st.error(response.explanation)
    else:
        st.markdown(response.explanation)
        calc = response.calculation or {}
        model_id = response.identification.model_id
        _render_charts(st, pd, model_id, calc)

    with st.expander("Inspect Full Agent Pipeline Trace (Identify → Extract → Tool → Calculate → Verify)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Stage 2 — Model Identification**")
            st.json(
                {
                    "model_id": response.identification.model_id,
                    "display_name": response.identification.display_name,
                    "confidence": response.identification.confidence,
                    "matched_signals": response.identification.matched_signals,
                    "candidate_scores": response.identification.candidate_scores,
                }
            )
            st.markdown("**Stage 4 — Selected Python Tool**")
            if response.tool_selection:
                st.code(
                    f"{response.tool_selection.tool_name}({json.dumps(_json_safe(response.tool_selection.arguments), indent=2)})",
                    language="python",
                )
            else:
                st.caption("No tool executed.")

        with c2:
            st.markdown("**Stage 3 — Extracted Inputs, Meanings & Assumptions**")
            if response.extraction.extracted_facts:
                fact_rows = [
                    {
                        "Parameter": f.parameter,
                        "Value": str(_json_safe(f.value)),
                        "Meaning": f.meaning,
                        "Matched Text": f.source_text,
                    }
                    for f in response.extraction.extracted_facts
                ]
                st.dataframe(pd.DataFrame(fact_rows), use_container_width=True, hide_index=True)
            st.json(
                {
                    "inputs": _json_safe(response.extraction.inputs),
                    "missing_fields": response.extraction.missing_fields,
                    "assumptions": response.extraction.assumptions,
                    "method": response.extraction.method,
                }
            )
            st.markdown("**Stage 5 — Structured Python Tool Output**")
            if response.calculation:
                st.json(response.calculation)
            else:
                st.caption("No calculation output.")

        if response.verification:
            st.markdown("**Stage 6 — Independent Mathematical Verification Checks**")
            check_rows = [
                {
                    "Status": "✅ Pass" if c.passed else "❌ Fail",
                    "Check": c.name,
                    "Details": c.details,
                }
                for c in response.verification.checks
            ]
            st.dataframe(pd.DataFrame(check_rows), use_container_width=True, hide_index=True)


def _render_charts(st: Any, pd: Any, model_id: str | None, calc: Dict[str, Any]) -> None:
    if not calc:
        return
    inter = calc.get("intermediate_calculations", {})
    final = calc.get("final_answer", {})
    val_in = calc.get("inputs_extracted", {})

    if model_id == "eoq" and final.get("eoq"):
        st.subheader("EOQ Cost Curve")
        q_opt = final["eoq"]
        D = float(val_in["annual_demand"])
        S = float(val_in["ordering_cost"])
        H = float(inter["holding_cost_per_unit"])
        qs = [q_opt * factor for factor in [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0]]
        df = pd.DataFrame(
            {
                "Order Quantity (Q)": [round(q, 1) for q in qs],
                "Ordering Cost ($)": [(D / q) * S for q in qs],
                "Holding Cost ($)": [(q / 2.0) * H for q in qs],
                "Total Relevant Cost ($)": [(D / q) * S + (q / 2.0) * H for q in qs],
            }
        ).set_index("Order Quantity (Q)")
        st.line_chart(df)

    elif model_id == "quantity_discount" and inter.get("tier_evaluations"):
        st.subheader("Total Annual Cost Comparison by Discount Tier")
        df = pd.DataFrame(
            [
                {
                    "Tier": f"Tier {t['tier']} (${t['unit_cost']:,.2f})",
                    "Total Annual Cost ($)": t["total_annual_cost"],
                }
                for t in inter["tier_evaluations"]
            ]
        ).set_index("Tier")
        st.bar_chart(df)

    elif model_id == "safety_stock_rop" and final.get("reorder_point") is not None:
        st.subheader("Reorder Point Breakdown")
        df = pd.DataFrame(
            {
                "Units": [
                    inter["expected_demand_during_lead_time"],
                    final["safety_stock"],
                    final["reorder_point"],
                ]
            },
            index=["Expected Lead-Time Demand", "Safety Stock", "Reorder Point (ROP)"],
        )
        st.bar_chart(df)

    elif model_id == "fixed_period_review" and final.get("target_inventory") is not None:
        st.subheader("Target Inventory & Order Quantity Breakdown")
        labels = ["Expected Demand (T+L)", "Safety Stock", "Target Inventory"]
        vals = [
            inter["expected_demand_protection_period"],
            final["safety_stock"],
            final["target_inventory"],
        ]
        if inter.get("inventory_position") is not None:
            labels.extend(["Inventory Position", "Recommended Order Qty"])
            vals.extend([inter["inventory_position"], final["order_quantity"]])
        df = pd.DataFrame({"Units": vals}, index=labels)
        st.bar_chart(df)

    elif model_id == "newsvendor" and final.get("recommended_order_qty") is not None:
        st.subheader("Expected Single-Period Volume Breakdown")
        df = pd.DataFrame(
            {
                "Units": [
                    val_in["mean_demand"],
                    final["recommended_order_qty"],
                    inter["expected_sales"],
                    inter["expected_shortage"],
                    inter["expected_leftover"],
                ]
            },
            index=[
                "Mean Demand",
                "Recommended Order Qty (Q*)",
                "Expected Sales",
                "Expected Lost Sales",
                "Expected Leftover",
            ],
        )
        st.bar_chart(df)

    elif model_id == "inventory_classification" and inter.get("class_summary"):
        st.subheader("Inventory Classification Breakdown (A, B, C, D)")
        summary_chart_df = pd.DataFrame(inter["class_summary"]).set_index("Final Inventory Class")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Item Count by Class**")
            st.bar_chart(summary_chart_df[["Item Count"]])
        with c2:
            st.markdown("**Total Sales ($) by Class**")
            st.bar_chart(summary_chart_df[["Total Sales ($)"]])


if __name__ == "__main__":
    if "--cli" in sys.argv:
        parser = argparse.ArgumentParser(description="Supply Chain Inventory Decision Agent CLI")
        parser.add_argument("--cli", action="store_true", help="Run in command-line mode")
        parser.add_argument("question", nargs="?", default=None, help="Optional single question")
        parser.add_argument("--llm", action="store_true", help="Enable optional OpenAI fallback")
        args = parser.parse_args()
        run_cli(question=args.question, use_llm=args.llm)
    else:
        _render_streamlit_app()
