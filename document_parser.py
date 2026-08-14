"""Extract structured financial inputs from credit documents via local Qwen LLM with intelligent page filtering."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import fitz
from openai import OpenAI
from pydantic import BaseModel, Field

DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.6:27b")

# Keywords used to identify pages containing core financial statements and debt covenants
DEFAULT_TARGET_KEYWORDS = [
    "balance sheet",
    "income statement",
    "total assets",
    "statement of financial position",
    "statement of operations",
    "cash flows",
    "covenant",
    "financial ratio",
    "leverage ratio",
    "interest coverage",
    "short-term debt",
    "long-term debt",
    "borrowings",
    "operating leases",
    "contingent liabilities",
    "retained earnings",
    "working capital",
]


class ExtractedCovenant(BaseModel):
    covenant_type: str = Field(
        description="Covenant name, e.g. 'Max Net Debt/EBITDA' or 'Min Interest Coverage'."
    )
    threshold_value: float = Field(
        description="Numeric threshold (ratio or multiple as stated in the document)."
    )
    page_number: int = Field(
        description="1-based page number where the covenant appears in the source PDF."
    )


class FinancialInputs(BaseModel):
    company_name: str = Field(description="Legal or reporting name of the borrower/issuer.")
    period_ending: str = Field(
        description="Reporting period end date as stated in the document (e.g. '2024-12-31')."
    )
    cash_and_equivalents: float = Field(description="Cash and cash equivalents.")
    working_capital: float = Field(description="Current assets minus current liabilities.")
    total_assets: float
    retained_earnings: float
    ebit: float = Field(description="Earnings before interest and taxes.")
    ebitda: float
    interest_expense: float
    market_cap_or_equity: float = Field(
        description="Market capitalization if public; book value of equity if private."
    )
    total_liabilities: float
    short_term_debt: float
    long_term_debt: float
    sales: float = Field(description="Revenue or net sales for the period.")
    extracted_covenants: list[ExtractedCovenant] = Field(default_factory=list)

    @property
    def net_debt(self) -> float:
        return (self.short_term_debt + self.long_term_debt) - self.cash_and_equivalents

    @property
    def default_point(self) -> float:
        """Merton model default point: ST debt plus half of LT debt (KMV convention)."""
        return self.short_term_debt + (0.5 * self.long_term_debt)


def extract_relevant_pdf_pages(
    pdf_file_path: str | Path,
    keywords: list[str] | None = None,
    max_fallback_pages: int = 15,
    status_callback: Callable[[str], None] | None = None,
) -> str:
    """Scan a PDF (digital or scanned image) and extract text from credit-relevant pages."""
    path = Path(pdf_file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    search_keywords = [kw.lower() for kw in (keywords or DEFAULT_TARGET_KEYWORDS)]
    matched_pages: list[str] = []
    all_pages: list[str] = []

    if status_callback:
        status_callback("📄 Opening PDF and scanning text layers / OCR...")

    with fitz.open(path) as doc:
        for page_number, page in enumerate(doc, start=1):
            # 1. Try normal text extraction
            text = page.get_text().strip()

            # 2. If text is empty, fallback to PyMuPDF's built-in OCR
            if not text:
                try:
                    ocr_textpage = page.get_textpage_ocr(flags=0, language="eng")
                    text = page.get_text(textpage=ocr_textpage).strip()
                except Exception:
                    pass

            if not text:
                continue

            page_header = f"--- Page {page_number} ---\n{text}"
            all_pages.append(page_header)

            text_lower = text.lower()
            if any(kw in text_lower for kw in search_keywords):
                matched_pages.append(page_header)

    if not matched_pages:
        if status_callback:
            status_callback("⚠️ No specific keyword matches found; using initial fallback pages.")
        return "\n\n".join(all_pages[:max_fallback_pages])

    if status_callback:
        status_callback(f"🔍 Found {len(matched_pages)} credit-relevant pages for analysis.")

    return "\n\n".join(matched_pages)


def parse_credit_document(
    pdf_file_path: str | Path,
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    status_callback: Callable[[str], None] | None = None,
) -> FinancialInputs:
    """Extract filtered text from a credit PDF and parse structured financial inputs via Qwen."""
    document_text = extract_relevant_pdf_pages(
        pdf_file_path, status_callback=status_callback
    )
    if not document_text.strip():
        raise ValueError(f"No extractable text in PDF: {pdf_file_path}")

    client = OpenAI(base_url=base_url, api_key="ollama")

    if status_callback:
        status_callback(f"🤖 Transmitting extracted context to local LLM ({model})...")

    completion = client.beta.chat.completions.parse(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a credit analyst extracting structured financial data from "
                    "loan agreements, credit memos, and annual reports. "
                    "Return all numeric fields in the same currency units as the source "
                    "(typically millions or thousands — be consistent with the document). "
                    "Use 0.0 for any metric not disclosed. "
                    "For extracted_covenants, list every financial covenant with its "
                    "threshold and the page number where it appears. "
                    "Normalize covenant_type to concise labels such as "
                    "'Max Net Debt/EBITDA' or 'Min Interest Coverage'."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Extract all financial inputs and covenants from the following document excerpt:\n\n"
                    f"{document_text}"
                ),
            },
        ],
        response_format=FinancialInputs,
    )

    if status_callback:
        status_callback("📊 Validating response with Pydantic schema...")

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("LLM returned no parsed FinancialInputs object.")

    return parsed