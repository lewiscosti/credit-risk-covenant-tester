"""Standard credit risk models: Altman Z-Score, Merton DD, and Monte Carlo covenant stress."""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
from scipy.optimize import root
from scipy.stats import norm


class AltmanZResult(TypedDict):
    z_score: float
    z_double_prime: float
    components: dict[str, float]
    z_score_zone: str
    z_double_prime_zone: str
    z_score_rating: str
    z_double_prime_rating: str


class MertonPDResult(TypedDict):
    asset_value: float
    asset_volatility: float
    distance_to_default: float
    probability_of_default_pct: float


class CashflowSimResult(TypedDict):
    ebitda_mean: float
    ebitda_median: float
    ebitda_std: float
    ebitda_p5: float
    ebitda_p95: float
    probability_of_breach_pct: float
    leverage_breach_pct: float
    coverage_breach_pct: float
    breach_severity: dict[str, float]


def _classify_altman_zone(z: float) -> str:
    if z > 2.99:
        return "Safe"
    if z >= 1.81:
        return "Grey"
    return "Distress"


def _altman_rating_equivalent(zone: str) -> str:
    return {
        "Safe": "BBB+ / Investment Grade",
        "Grey": "BB to B / Speculative",
        "Distress": "CCC and below / High Default Risk",
    }[zone]


def calculate_altman_z(
    working_capital: float,
    total_assets: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_liabilities: float,
    sales: float,
) -> AltmanZResult:
    """Compute Altman Z-Score (manufacturing) and Z''-Score (non-manufacturing / private).

    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Value of Equity / Total Liabilities  (Z) or Book Equity / Total Liabilities (Z'')
    X5 = Sales / Total Assets
    """
    if total_assets <= 0:
        raise ValueError("total_assets must be positive")
    if total_liabilities <= 0:
        raise ValueError("total_liabilities must be positive")

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4_market = market_cap / total_liabilities
    x5 = sales / total_assets

    book_equity = total_assets - total_liabilities
    x4_book = book_equity / total_liabilities

    z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4_market + 0.99 * x5
    z_double_prime = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4_book

    z_zone = _classify_altman_zone(z_score)
    zdp_zone = _classify_altman_zone(z_double_prime)

    return {
        "z_score": z_score,
        "z_double_prime": z_double_prime,
        "components": {
            "X1_working_capital_to_assets": x1,
            "X2_retained_earnings_to_assets": x2,
            "X3_ebit_to_assets": x3,
            "X4_market_equity_to_liabilities": x4_market,
            "X4_book_equity_to_liabilities": x4_book,
            "X5_sales_to_assets": x5,
        },
        "z_score_zone": z_zone,
        "z_double_prime_zone": zdp_zone,
        "z_score_rating": _altman_rating_equivalent(z_zone),
        "z_double_prime_rating": _altman_rating_equivalent(zdp_zone),
    }


def _merton_d1_d2(
    asset_value: float,
    total_debt: float,
    asset_volatility: float,
    risk_free_rate: float,
    time_horizon: float,
) -> tuple[float, float]:
    sqrt_t = math.sqrt(time_horizon)
    d1 = (
        math.log(asset_value / total_debt)
        + (risk_free_rate + 0.5 * asset_volatility**2) * time_horizon
    ) / (asset_volatility * sqrt_t)
    d2 = d1 - asset_volatility * sqrt_t
    return d1, d2


def calculate_merton_pd(
    equity_value: float,
    equity_volatility: float,
    total_debt: float,
    risk_free_rate: float,
    time_horizon: float = 1.0,
) -> MertonPDResult:
    """Solve Merton structural model for asset value and volatility, then compute DD and PD."""
    if equity_value <= 0:
        raise ValueError("equity_value must be positive")
    if equity_volatility <= 0:
        raise ValueError("equity_volatility must be positive")
    if total_debt <= 0:
        raise ValueError("total_debt must be positive")
    if time_horizon <= 0:
        raise ValueError("time_horizon must be positive")

    def equations(vars_: np.ndarray) -> np.ndarray:
        v_a, sigma_a = vars_
        if v_a <= 0 or sigma_a <= 0:
            return np.array([1e6, 1e6])

        d1, d2 = _merton_d1_d2(v_a, total_debt, sigma_a, risk_free_rate, time_horizon)
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)

        eq1 = v_a * nd1 - total_debt * math.exp(-risk_free_rate * time_horizon) * nd2 - equity_value
        eq2 = sigma_a * v_a * nd1 - equity_volatility * equity_value
        return np.array([eq1, eq2])

    v_a_guess = equity_value + total_debt
    sigma_a_guess = equity_volatility * equity_value / v_a_guess
    solution = root(equations, x0=np.array([v_a_guess, sigma_a_guess]), method="hybr")

    if not solution.success:
        raise RuntimeError(f"Merton solver failed to converge: {solution.message}")

    asset_value, asset_volatility = solution.x
    sqrt_t = math.sqrt(time_horizon)
    distance_to_default = (
        math.log(asset_value / total_debt)
        + (risk_free_rate - 0.5 * asset_volatility**2) * time_horizon
    ) / (asset_volatility * sqrt_t)
    probability_of_default = norm.cdf(-distance_to_default)

    return {
        "asset_value": float(asset_value),
        "asset_volatility": float(asset_volatility),
        "distance_to_default": float(distance_to_default),
        "probability_of_default_pct": float(probability_of_default * 100),
    }


def run_cashflow_simulation(
    base_ebitda: float,
    ebitda_volatility: float,
    annual_debt_service: float,
    max_leverage_covenant: float,
    min_coverage_covenant: float,
    net_debt: float,
    num_simulations: int = 5000,
    random_seed: int | None = None,
) -> CashflowSimResult:
    """Monte Carlo 1-year EBITDA stress test against leverage and coverage covenants.

    EBITDA paths are drawn from a log-normal distribution. ``annual_debt_service`` is
    treated as annual interest expense for the coverage ratio (EBITDA / Interest).
    """
    if base_ebitda <= 0:
        raise ValueError("base_ebitda must be positive")
    if ebitda_volatility < 0:
        raise ValueError("ebitda_volatility must be non-negative")
    if max_leverage_covenant <= 0:
        raise ValueError("max_leverage_covenant must be positive")
    if min_coverage_covenant <= 0:
        raise ValueError("min_coverage_covenant must be positive")
    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive")

    rng = np.random.default_rng(random_seed)
    shocks = rng.standard_normal(num_simulations)
    simulated_ebitda = base_ebitda * np.exp(
        -0.5 * ebitda_volatility**2 + ebitda_volatility * shocks
    )

    leverage_ratio = net_debt / simulated_ebitda
    coverage_ratio = simulated_ebitda / annual_debt_service

    leverage_breach = leverage_ratio > max_leverage_covenant
    coverage_breach = coverage_ratio < min_coverage_covenant
    any_breach = leverage_breach | coverage_breach

    leverage_excess = np.maximum(leverage_ratio - max_leverage_covenant, 0.0)
    coverage_shortfall = np.maximum(min_coverage_covenant - coverage_ratio, 0.0)

    breached = any_breach
    if breached.any():
        avg_leverage_excess = float(leverage_excess[breached].mean())
        avg_coverage_shortfall = float(coverage_shortfall[breached].mean())
    else:
        avg_leverage_excess = 0.0
        avg_coverage_shortfall = 0.0

    return {
        "ebitda_mean": float(simulated_ebitda.mean()),
        "ebitda_median": float(np.median(simulated_ebitda)),
        "ebitda_std": float(simulated_ebitda.std()),
        "ebitda_p5": float(np.percentile(simulated_ebitda, 5)),
        "ebitda_p95": float(np.percentile(simulated_ebitda, 95)),
        "probability_of_breach_pct": float(any_breach.mean() * 100),
        "leverage_breach_pct": float(leverage_breach.mean() * 100),
        "coverage_breach_pct": float(coverage_breach.mean() * 100),
        "breach_severity": {
            "avg_leverage_excess_on_breach": avg_leverage_excess,
            "avg_coverage_shortfall_on_breach": avg_coverage_shortfall,
            "max_leverage_excess": float(leverage_excess.max()),
            "max_coverage_shortfall": float(coverage_shortfall.max()),
        },
    }
