"""Formal 2-page Credit Recommendation Memorandum PDF generator."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, TypedDict

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from document_parser import FinancialInputs

PAGE_MARGIN = 36  # 0.5 inch
CONTENT_WIDTH = letter[0] - (2 * PAGE_MARGIN)


class StressScenario(TypedDict):
    label: str
    net_debt_ebitda: float
    interest_coverage: float
    breach_probability_pct: float


class StressResults(TypedDict, total=False):
    scenarios: list[StressScenario]


def _escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _cell(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_escape(text), style)


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "memo_title": ParagraphStyle(
            "MemoTitle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=16,
            spaceAfter=4,
            textColor=colors.HexColor("#1e3a5f"),
        ),
        "section_header": ParagraphStyle(
            "SectionHeader",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            spaceBefore=4,
            spaceAfter=4,
            textColor=colors.HexColor("#1e3a5f"),
        ),
        "body_dark": ParagraphStyle(
            "BodyDark",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#1f2937"),
        ),
        "body_light": ParagraphStyle(
            "BodyLight",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#4b5563"),
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#111827"),
        ),
        "badge_approved": ParagraphStyle(
            "BadgeApproved",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#065f46"),
            backColor=colors.HexColor("#d1fae5"),
            borderPadding=6,
        ),
        "badge_declined": ParagraphStyle(
            "BadgeDeclined",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#991b1b"),
            backColor=colors.HexColor("#fee2e2"),
            borderPadding=6,
        ),
        "bullet": ParagraphStyle(
            "BulletItem",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            leftIndent=12,
            bulletIndent=0,
            textColor=colors.HexColor("#374151"),
        ),
    }


def _table_style_header() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )


def _table_style_grid() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )


def _recommendation_badge(
    z_results: dict[str, Any],
    stress_results: StressResults | dict[str, Any],
) -> tuple[str, str]:
    scenarios = stress_results.get("scenarios", [])
    base_breach = scenarios[0]["breach_probability_pct"] if scenarios else 100.0
    zone = z_results.get("z_score_zone", "Grey")

    approved = zone == "Safe" and base_breach < 20.0
    label = "APPROVED" if approved else "DECLINED"
    return label, "badge_approved" if approved else "badge_declined"


def _risk_disclosures(inputs: FinancialInputs) -> list[str]:
    disclosures: list[str] = []
    for covenant in inputs.extracted_covenants:
        disclosures.append(
            f"{covenant.covenant_type} covenant (threshold: {covenant.threshold_value:g}) "
            f"— source document, p. {covenant.page_number}."
        )
    if not disclosures:
        disclosures.append(
            "No explicit financial covenants extracted; stress-test thresholds applied "
            "from institutional defaults."
        )
    disclosures.append(
        "Merton structural default metrics assume estimated equity volatility and KMV "
        "default-point debt aggregation (ST debt + 50% LT debt)."
    )
    disclosures.append(
        "Monte Carlo covenant breach probabilities reflect 5,000 simulated 1-year "
        "log-normal EBITDA paths."
    )
    return disclosures


def generate_credit_memo(
    inputs: FinancialInputs,
    z_results: dict[str, Any],
    merton_results: dict[str, Any],
    stress_results: StressResults | dict[str, Any],
    summary_text: str,
) -> bytes:
    """Build a formal 2-page Credit Recommendation Memorandum PDF."""
    styles = _build_styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN,
    )

    today = datetime.now(tz=timezone.utc).strftime("%B %d, %Y")
    net_debt_ebitda = inputs.net_debt / inputs.ebitda if inputs.ebitda else 0.0
    risk_rating = z_results.get("z_score_rating", "N/A")
    badge_label, badge_style_key = _recommendation_badge(z_results, stress_results)

    col2 = CONTENT_WIDTH / 2
    header_data = [
        [
            _cell("MEMORANDUM TO: Credit Committee", styles["body_dark"]),
            _cell(f"DATE: {today}", styles["body_dark"]),
        ],
        [
            _cell("FROM: Senior Risk Analyst", styles["body_dark"]),
            _cell(f"BORROWER: {inputs.company_name}", styles["body_dark"]),
        ],
        [
            _cell("REQUESTED FACILITY: Senior Secured Term Loan", styles["body_dark"]),
            _cell(f"ASSIGNED RISK RATING: {risk_rating}", styles["body_dark"]),
        ],
    ]
    header_table = Table(header_data, colWidths=[col2, col2])
    header_table.setStyle(_table_style_grid())

    badge = Paragraph(f"<b>{badge_label}</b>", styles[badge_style_key])

    metrics_data = [
        [
            _cell("Metric", styles["table_header"]),
            _cell("Value", styles["table_header"]),
        ],
        [
            _cell("Altman Z-Score", styles["table_cell"]),
            _cell(
                f"{z_results['z_score']:.2f} ({z_results['z_score_zone']})",
                styles["table_cell"],
            ),
        ],
        [
            _cell("Merton Distance-to-Default", styles["table_cell"]),
            _cell(
                f"{merton_results['distance_to_default']:.2f}σ", styles["table_cell"]
            ),
        ],
        [
            _cell("Probability of Default (1Y)", styles["table_cell"]),
            _cell(
                f"{merton_results['probability_of_default_pct']:.2f}%",
                styles["table_cell"],
            ),
        ],
        [
            _cell("Net Debt / EBITDA", styles["table_cell"]),
            _cell(f"{net_debt_ebitda:.2f}x", styles["table_cell"]),
        ],
    ]
    metrics_table = Table(metrics_data, colWidths=[col2, col2])
    metrics_table.setStyle(_table_style_header())

    scenarios: list[StressScenario] = stress_results.get("scenarios", [])
    stress_header = [
        _cell("Scenario", styles["table_header"]),
        _cell("Net Debt / EBITDA", styles["table_header"]),
        _cell("Interest Coverage", styles["table_header"]),
        _cell("Covenant Breach Prob.", styles["table_header"]),
    ]
    stress_rows = [stress_header]
    for scenario in scenarios:
        stress_rows.append(
            [
                _cell(scenario["label"], styles["table_cell"]),
                _cell(f"{scenario['net_debt_ebitda']:.2f}x", styles["table_cell"]),
                _cell(f"{scenario['interest_coverage']:.2f}x", styles["table_cell"]),
                _cell(
                    f"{scenario['breach_probability_pct']:.1f}%", styles["table_cell"]
                ),
            ]
        )
    stress_col = CONTENT_WIDTH / 4
    stress_table = Table(stress_rows, colWidths=[stress_col] * 4)
    stress_table.setStyle(_table_style_header())

    footnotes = _risk_disclosures(inputs)
    footnote_flowables = [
        Paragraph(f"• {_escape(item)}", styles["bullet"]) for item in footnotes
    ]

    story = [
        Paragraph("CREDIT RECOMMENDATION MEMORANDUM", styles["memo_title"]),
        Spacer(1, 8),
        header_table,
        Spacer(1, 10),
        Paragraph(
            "Section 1 — Executive Summary &amp; Recommendation",
            styles["section_header"],
        ),
        badge,
        Spacer(1, 6),
        Paragraph(_escape(summary_text), styles["body_dark"]),
        Spacer(1, 10),
        Paragraph(
            "Section 2 — Balance Sheet &amp; Structural Credit Metrics",
            styles["section_header"],
        ),
        metrics_table,
        Spacer(1, 10),
        Paragraph(
            "Section 3 — Covenant Stress-Test Sensitivity Matrix",
            styles["section_header"],
        ),
        stress_table,
        Spacer(1, 10),
        Paragraph(
            "Section 4 — Key Risk Disclosures &amp; Footnotes", styles["section_header"]
        ),
        *footnote_flowables,
        Spacer(1, 8),
        Paragraph(
            _escape(
                f"Reporting period: {inputs.period_ending}. "
                f"Figures in source-document units. Confidential — Credit Committee use only."
            ),
            styles["body_light"],
        ),
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
