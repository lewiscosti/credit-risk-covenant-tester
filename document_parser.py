"""Extract structured financial inputs from credit documents via local Qwen LLM with intelligent page filtering."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

import fitz
from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen-3.8-instruct")

# Sampling parameters for the extraction call
EXTRACTION_TEMPERATURE = 0.7
EXTRACTION_TOP_P = 0.8
EXTRACTION_PRESENCE_PENALTY = 1.5

# Ollama-specific options, passed via extra_body to the OpenAI-compatible endpoint.
# - num_ctx: 128k token context window so large documents are not truncated
# - num_predict: 32k max output tokens so the JSON response is not cut off
# - chat_template_kwargs: disables Qwen3 "thinking" mode (it breaks structured output)
OLLAMA_EXTRA_BODY = {
    "top_k": 20,
    "num_ctx": 131072,
    "num_predict": 32768,
    "chat_template_kwargs": {"enable_thinking": False},
}

client = OpenAI(
    base_url=DEFAULT_BASE_URL,
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
)

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
    company_name: str = Field(
        description="Legal or reporting name of the borrower/issuer."
    )
    period_ending: str = Field(
        description="Reporting period end date as stated in the document (e.g. '2024-12-31')."
    )
    cash_and_equivalents: float | None = Field(
        default=None, description="Cash and cash equivalents."
    )
    working_capital: float | None = Field(
        default=None, description="Current assets minus current liabilities."
    )
    total_assets: float | None = None
    retained_earnings: float | None = None
    ebit: float | None = Field(
        default=None, description="Earnings before interest and taxes."
    )
    ebitda: float | None = None
    interest_expense: float | None = None
    market_cap_or_equity: float | None = Field(
        default=None,
        description="Market capitalization if public; book value of equity if private.",
    )
    total_liabilities: float | None = None
    short_term_debt: float | None = None
    long_term_debt: float | None = None
    sales: float | None = Field(
        default=None, description="Revenue or net sales for the period."
    )
    extracted_covenants: list[ExtractedCovenant] = Field(default_factory=list)

    @property
    def net_debt(self) -> float:
        st = self.short_term_debt or 0.0
        lt = self.long_term_debt or 0.0
        cash = self.cash_and_equivalents or 0.0
        return (st + lt) - cash

    @property
    def default_point(self) -> float:
        """Merton model default point: ST debt plus half of LT debt (KMV convention)."""
        st = self.short_term_debt or 0.0
        lt = self.long_term_debt or 0.0
        return st + (0.5 * lt)

    def sanitize(self) -> list[str]:
        """Resolve None and non-positive values using accounting identities.

        Returns a list of warnings describing each fallback applied.
        Uses A = L + E to derive missing total_assets or total_liabilities.
        """
        warnings: list[str] = []

        # Phase 1 — Resolve the A / L / E triangle
        ta, tl, eq = (
            self.total_assets,
            self.total_liabilities,
            self.market_cap_or_equity,
        )

        if ta is None and tl is not None and eq is not None:
            self.total_assets = max(tl + eq, 1.0)
            warnings.append(
                f"total_assets was null; estimated as total_liabilities ({tl}) + equity ({eq})."
            )
        elif tl is None and ta is not None and eq is not None:
            self.total_liabilities = max(ta - eq, 1.0)
            warnings.append(
                f"total_liabilities was null; estimated as total_assets ({ta}) - equity ({eq})."
            )
        elif ta is None and tl is None:
            # Both missing — use minimal fallbacks
            self.total_assets = 1.0
            self.total_liabilities = 1.0
            warnings.append(
                "Both total_assets and total_liabilities were null; using minimal fallback values."
            )

        # Ensure positivity after phase 1
        if (self.total_assets or 0) <= 0:
            self.total_assets = 1.0
            warnings.append("total_assets was non-positive; clamped to 1.0.")
        if (self.total_liabilities or 0) <= 0:
            self.total_liabilities = 1.0
            warnings.append("total_liabilities was non-positive; clamped to 1.0.")

        # Phase 2 — Default remaining None fields to 0.0
        for field in (
            "cash_and_equivalents",
            "working_capital",
            "retained_earnings",
            "ebit",
            "ebitda",
            "interest_expense",
            "market_cap_or_equity",
            "short_term_debt",
            "long_term_debt",
            "sales",
        ):
            if getattr(self, field) is None:
                setattr(self, field, 0.0)
                warnings.append(f"{field} was null; defaulted to 0.0.")

        return warnings


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
        for page_number in range(1, doc.page_count + 1):
            page = doc.load_page(page_number - 1)

            # 1. Try normal text extraction
            # (cast: PyMuPDF's untyped get_text() returns str for the default "text" option)
            text = cast(str, page.get_text()).strip()

            # 2. If text is empty, fallback to PyMuPDF's built-in OCR
            if not text:
                try:
                    ocr_textpage = page.get_textpage_ocr(flags=0, language="eng")
                    text = cast(str, page.get_text(textpage=ocr_textpage)).strip()
                except (RuntimeError, ValueError) as exc:
                    logger.debug(
                        "OCR fallback failed for page %d: %s", page_number, exc
                    )

            if not text:
                continue

            page_header = f"--- Page {page_number} ---\n{text}"
            all_pages.append(page_header)

            text_lower = text.lower()
            if any(kw in text_lower for kw in search_keywords):
                matched_pages.append(page_header)

    if not matched_pages:
        if status_callback:
            status_callback(
                "⚠️ No specific keyword matches found; using initial fallback pages."
            )
        return "\n\n".join(all_pages[:max_fallback_pages])

    if status_callback:
        status_callback(
            f"🔍 Found {len(matched_pages)} credit-relevant pages for analysis."
        )

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
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a credit analyst extracting structured financial data from "
                    "loan agreements, credit memos, and annual reports. "
                    "Return all numeric fields in the same currency units as the source "
                    "(typically millions or thousands — be consistent with the document). "
                    "Use null for any metric not disclosed or unable to be determined. "
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
        temperature=EXTRACTION_TEMPERATURE,
        top_p=EXTRACTION_TOP_P,
        presence_penalty=EXTRACTION_PRESENCE_PENALTY,
        extra_body=OLLAMA_EXTRA_BODY,
    )

    if status_callback:
        status_callback("📊 Validating response with Pydantic schema...")

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("LLM returned no parsed FinancialInputs object.")

    return parsed
