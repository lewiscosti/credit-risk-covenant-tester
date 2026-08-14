"""Streamlit dashboard for credit risk and covenant stress testing."""

from __future__ import annotations

import io
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from credit_math import calculate_altman_z, calculate_merton_pd, run_cashflow_simulation
from document_parser import FinancialInputs, parse_credit_document

DEFAULT_EQUITY_VOLATILITY = 0.35
DEFAULT_RISK_FREE_RATE = 0.04
DEFAULT_EBITDA_VOLATILITY = 0.25
DEFAULT_MAX_LEVERAGE = 4.0
DEFAULT_MIN_COVERAGE = 2.0


def _parse_covenant_thresholds(
    covenants: list,
) -> tuple[float, float]:
    max_leverage = DEFAULT_MAX_LEVERAGE
    min_coverage = DEFAULT_MIN_COVERAGE

    for covenant in covenants:
        label = covenant.covenant_type.lower()
        if any(k in label for k in ("leverage", "debt/ebitda", "net debt")):
            max_leverage = covenant.threshold_value
        elif any(k in label for k in ("coverage", "interest")):
            min_coverage = covenant.threshold_value

    return max_leverage, min_coverage


def _simulate_ebitda_paths(
    base_ebitda: float,
    ebitda_volatility: float,
    num_simulations: int = 5000,
    random_seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    shocks = rng.standard_normal(num_simulations)
    return base_ebitda * np.exp(-0.5 * ebitda_volatility**2 + ebitda_volatility * shocks)


def _stressed_interest(base_interest: float, rate_hike_bps: int) -> float:
    return base_interest * (1 + rate_hike_bps / 10_000)


def _build_credit_memo_text(
    data: FinancialInputs,
    altman: dict,
    merton: dict,
    base_sim: dict,
    stressed_sim: dict,
    haircut_pct: int,
    rate_hike_bps: int,
) -> str:
    leverage = data.net_debt / data.ebitda if data.ebitda else float("inf")
    return f"""CREDIT MEMORANDUM — {data.company_name.upper()}
Period Ending: {data.period_ending}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

EXECUTIVE SUMMARY
{data.company_name} presents a {altman['z_score_zone']} credit profile based on an Altman Z-Score
of {altman['z_score']:.2f} ({altman['z_score_rating']}). Structural default risk per the Merton
model implies a {merton['probability_of_default_pct']:.2f}% probability of default over a
one-year horizon (Distance-to-Default: {merton['distance_to_default']:.2f}σ).

KEY METRICS
  Altman Z-Score:          {altman['z_score']:.2f}  [{altman['z_score_zone']}]
  Merton Distance-to-Def:  {merton['distance_to_default']:.2f}
  Merton PD (1Y):          {merton['probability_of_default_pct']:.2f}%
  Net Debt / EBITDA:       {leverage:.2f}x
  Net Debt:                {data.net_debt:,.0f}
  EBITDA:                  {data.ebitda:,.0f}

COVENANT STRESS TEST
Macro assumptions — EBITDA haircut: {haircut_pct}%; rate hike: +{rate_hike_bps} bps.

  Base Case Breach Probability:     {base_sim['probability_of_breach_pct']:.1f}%
    Leverage breaches:              {base_sim['leverage_breach_pct']:.1f}%
    Coverage breaches:              {base_sim['coverage_breach_pct']:.1f}%

  Stressed Case Breach Probability: {stressed_sim['probability_of_breach_pct']:.1f}%
    Leverage breaches:              {stressed_sim['leverage_breach_pct']:.1f}%
    Coverage breaches:              {stressed_sim['coverage_breach_pct']:.1f}%

RECOMMENDATION
{"Maintain exposure with standard monitoring." 
    if stressed_sim['probability_of_breach_pct'] < 15 and altman['z_score_zone'] != "Distress"
    else "Enhanced monitoring recommended — structural accounting distress or elevated covenant risk."
    if stressed_sim['probability_of_breach_pct'] < 35
    else "Consider risk mitigation — material deterioration under macro stress scenario."}

Extracted Covenants: {len(data.extracted_covenants)} identified in source document.
"""


def _build_credit_memo_pdf(memo_text: str, company_name: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"<b>Credit Memorandum — {company_name}</b>", styles["Title"]),
        Spacer(1, 12),
    ]
    for line in memo_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line.replace("  ", "&nbsp;&nbsp;"), styles["Normal"]))
        else:
            story.append(Spacer(1, 6))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def _metric_card(label: str, value: str, delta: str | None = None) -> None:
    st.metric(label=label, value=value, delta=delta)


def main() -> None:
    st.set_page_config(
        page_title="Credit Risk & Covenant Tester",
        page_icon="📊",
        layout="wide",
    )
    st.title("Credit Risk & Covenant Tester")

    with st.sidebar:
        st.header("Inputs")
        uploaded_pdf = st.file_uploader("Upload Credit PDF", type=["pdf"])
        api_base_url = st.text_input("API Base URL", value="http://localhost:11434/v1")

        st.header("Macro Stress Controls")
        ebitda_haircut_pct = st.slider(
            "EBITDA Downside Haircut %",
            min_value=0,
            max_value=50,
            value=0,
            step=5,
        )
        rate_hike_bps = st.slider(
            "Interest Rate Hike (bps)",
            min_value=0,
            max_value=500,
            value=0,
            step=25,
        )

    if uploaded_pdf is None:
        st.info("Upload a credit PDF in the sidebar to begin analysis.")
        return

    if st.button("Analyze Document", type="primary"):
        with st.status("Initializing Document Pipeline...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded_pdf.getvalue())
                tmp_path = Path(tmp.name)

            try:
                st.session_state["financial_inputs"] = parse_credit_document(
                    tmp_path,
                    base_url=api_base_url,
                    status_callback=status.write,
                )
                status.update(
                    label="Analysis Complete!", state="complete", expanded=False
                )
            except Exception as e:
                status.update(
                    label="Extraction Failed!", state="error", expanded=True
                )
                st.error(f"Error parsing credit document: {e}")
                return
            finally:
                tmp_path.unlink(missing_ok=True)

    if "financial_inputs" not in st.session_state:
        st.info("Click **Analyze Document** to run extraction.")
        return

    data: FinancialInputs = st.session_state["financial_inputs"]
    max_leverage, min_coverage = _parse_covenant_thresholds(data.extracted_covenants)

    # Sanitize total_assets to handle OCR gaps in scanned PDFs
    total_assets = data.total_assets
    if total_assets <= 0:
        st.warning(
            "⚠️ **OCR Extraction Warning:** `total_assets` was extracted as `0.0`. "
            "Estimating total assets from total liabilities and equity to calculate Altman Z-Score."
        )
        total_assets = max(data.total_liabilities + data.market_cap_or_equity, 1.0)

    altman = calculate_altman_z(
        working_capital=data.working_capital,
        total_assets=total_assets,
        retained_earnings=data.retained_earnings,
        ebit=data.ebit,
        market_cap=data.market_cap_or_equity,
        total_liabilities=data.total_liabilities,
        sales=data.sales,
    )

    merton = calculate_merton_pd(
        equity_value=max(data.market_cap_or_equity, 1.0),
        equity_volatility=DEFAULT_EQUITY_VOLATILITY,
        total_debt=max(data.default_point, 1.0),
        risk_free_rate=DEFAULT_RISK_FREE_RATE + rate_hike_bps / 10_000,
    )

    base_ebitda = data.ebitda
    stressed_ebitda = data.ebitda * (1 - ebitda_haircut_pct / 100)
    base_interest = max(data.interest_expense, 1.0)
    stressed_interest = max(_stressed_interest(base_interest, rate_hike_bps), 1.0)
    net_debt = data.net_debt

    base_sim = run_cashflow_simulation(
        base_ebitda=max(base_ebitda, 1.0),
        ebitda_volatility=DEFAULT_EBITDA_VOLATILITY,
        annual_debt_service=base_interest,
        max_leverage_covenant=max_leverage,
        min_coverage_covenant=min_coverage,
        net_debt=net_debt,
    )

    stressed_sim = run_cashflow_simulation(
        base_ebitda=max(stressed_ebitda, 1.0),
        ebitda_volatility=DEFAULT_EBITDA_VOLATILITY,
        annual_debt_service=stressed_interest,
        max_leverage_covenant=max_leverage,
        min_coverage_covenant=min_coverage,
        net_debt=net_debt,
    )

    net_debt_to_ebitda = net_debt / base_ebitda if base_ebitda else float("inf")
    memo_text = _build_credit_memo_text(
        data, altman, merton, base_sim, stressed_sim, ebitda_haircut_pct, rate_hike_bps
    )

    tab_overview, tab_stress, tab_memo = st.tabs(
        ["Credit Risk Overview", "Covenant Stress Testing", "Credit Memo Preview & Download"]
    )

    with tab_overview:
        st.subheader(f"{data.company_name} — {data.period_ending}")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _metric_card("Altman Z-Score", f"{altman['z_score']:.2f}", altman["z_score_zone"])
        with c2:
            _metric_card("Merton DD", f"{merton['distance_to_default']:.2f}σ")
        with c3:
            _metric_card("Merton PD", f"{merton['probability_of_default_pct']:.2f}%")
        with c4:
            _metric_card("Net Debt / EBITDA", f"{net_debt_to_ebitda:.2f}x")

        st.caption(f"Rating equivalent: {altman['z_score_rating']}")

        with st.expander("Altman Components"):
            st.dataframe(
                pd.DataFrame(
                    [{"Component": k, "Value": f"{v:.4f}"} for k, v in altman["components"].items()]
                ),
                hide_index=True,
                width='stretch',
            )

    with tab_stress:
        st.subheader("Monte Carlo EBITDA Simulation (Stressed Case)")
        sim_paths = _simulate_ebitda_paths(
            base_ebitda=max(stressed_ebitda, 1.0),
            ebitda_volatility=DEFAULT_EBITDA_VOLATILITY,
        )

        # Restrain plot width by centering it in a sub-column
        _, col_chart, _ = st.columns([1, 2, 1])

        with col_chart:
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="none")
            ax.set_facecolor("none")

            ax.hist(
                sim_paths,
                bins=40,
                color="#3b82f6",
                edgecolor="#1e293b",
                alpha=0.9,
            )
            ax.axvline(
                stressed_ebitda,
                color="#ef4444",
                linestyle="--",
                linewidth=2,
                label="Stressed EBITDA",
            )

            ax.set_xlabel("Simulated 1Y EBITDA", color="white", fontsize=9)
            ax.set_ylabel("Frequency", color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=8)

            # Clean up plot borders to fit dark UI
            ax.spines["bottom"].set_color("#475569")
            ax.spines["left"].set_color("#475569")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.legend(
                facecolor="#0f172a",
                edgecolor="#334155",
                labelcolor="white",
                fontsize=8,
            )
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Covenant Breach Summary")
        breach_df = pd.DataFrame(
            {
                "Metric": [
                    "Overall Breach Probability",
                    "Leverage Breach Rate",
                    "Coverage Breach Rate",
                    "EBITDA Mean (simulated)",
                    "EBITDA P5",
                    "EBITDA P95",
                ],
                "Base Case": [
                    f"{base_sim['probability_of_breach_pct']:.1f}%",
                    f"{base_sim['leverage_breach_pct']:.1f}%",
                    f"{base_sim['coverage_breach_pct']:.1f}%",
                    f"{base_sim['ebitda_mean']:,.0f}",
                    f"{base_sim['ebitda_p5']:,.0f}",
                    f"{base_sim['ebitda_p95']:,.0f}",
                ],
                "Stressed Case": [
                    f"{stressed_sim['probability_of_breach_pct']:.1f}%",
                    f"{stressed_sim['leverage_breach_pct']:.1f}%",
                    f"{stressed_sim['coverage_breach_pct']:.1f}%",
                    f"{stressed_sim['ebitda_mean']:,.0f}",
                    f"{stressed_sim['ebitda_p5']:,.0f}",
                    f"{stressed_sim['ebitda_p95']:,.0f}",
                ],
            }
        )
        st.dataframe(breach_df, hide_index=True, width='stretch')

        if data.extracted_covenants:
            st.caption("Extracted covenant thresholds")
            st.dataframe(
                pd.DataFrame([c.model_dump() for c in data.extracted_covenants]),
                hide_index=True,
                width='stretch',
            )

    with tab_memo:
        st.subheader("Credit Recommendation Preview")
        st.text(memo_text)

        pdf_bytes = _build_credit_memo_pdf(memo_text, data.company_name)
        st.download_button(
            label="Download Credit Memo (PDF)",
            data=pdf_bytes,
            file_name=f"credit_memo_{data.company_name.replace(' ', '_')}.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()