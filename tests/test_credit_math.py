import pytest

from credit_math import (
    calculate_altman_z,
    calculate_merton_pd,
    run_cashflow_simulation,
)


class TestAltmanZScore:
    def test_altman_returns_expected_components(self):
        result = calculate_altman_z(
            working_capital=200,
            total_assets=1000,
            retained_earnings=300,
            ebit=100,
            market_cap=1200,
            total_liabilities=500,
            sales=1500,
        )

        assert result["components"]["X1_working_capital_to_assets"] == pytest.approx(0.20)
        assert result["components"]["X2_retained_earnings_to_assets"] == pytest.approx(0.30)
        assert result["components"]["X3_ebit_to_assets"] == pytest.approx(0.10)
        assert result["components"]["X4_market_equity_to_liabilities"] == pytest.approx(2.40)
        assert result["components"]["X5_sales_to_assets"] == pytest.approx(1.50)

    def test_altman_z_score_calculation(self):
        result = calculate_altman_z(
            working_capital=200,
            total_assets=1000,
            retained_earnings=300,
            ebit=100,
            market_cap=1200,
            total_liabilities=500,
            sales=1500,
        )

        expected_z = (
            1.2 * 0.20
            + 1.4 * 0.30
            + 3.3 * 0.10
            + 0.6 * 2.40
            + 0.99 * 1.50
        )

        assert result["z_score"] == pytest.approx(expected_z)

    def test_altman_z_double_prime_calculation(self):
        result = calculate_altman_z(
            working_capital=200,
            total_assets=1000,
            retained_earnings=300,
            ebit=100,
            market_cap=1200,
            total_liabilities=500,
            sales=1500,
        )

        book_equity = 1000 - 500
        x4_book = book_equity / 500

        expected_z_double_prime = (
            6.56 * 0.20
            + 3.26 * 0.30
            + 6.72 * 0.10
            + 1.05 * x4_book
        )

        assert result["z_double_prime"] == pytest.approx(expected_z_double_prime)

    def test_altman_safe_zone_classification(self):
        result = calculate_altman_z(
            working_capital=500,
            total_assets=1000,
            retained_earnings=500,
            ebit=200,
            market_cap=2000,
            total_liabilities=500,
            sales=1500,
        )

        assert result["z_score_zone"] == "Safe"

    def test_altman_invalid_total_assets(self):
        with pytest.raises(ValueError, match="total_assets must be positive"):
            calculate_altman_z(
                working_capital=100,
                total_assets=0,
                retained_earnings=100,
                ebit=50,
                market_cap=500,
                total_liabilities=200,
                sales=500,
            )

    def test_altman_invalid_total_liabilities(self):
        with pytest.raises(ValueError, match="total_liabilities must be positive"):
            calculate_altman_z(
                working_capital=100,
                total_assets=1000,
                retained_earnings=100,
                ebit=50,
                market_cap=500,
                total_liabilities=0,
                sales=500,
            )


class TestMertonModel:
    def test_merton_invalid_equity_volatility(self):
        with pytest.raises(ValueError, match="equity_volatility must be positive"):
            calculate_merton_pd(
            equity_value=500,
            equity_volatility=0,
            total_debt=400,
            risk_free_rate=0.04,
        )


    def test_merton_invalid_time_horizon(self):
        with pytest.raises(ValueError, match="time_horizon must be positive"):
            calculate_merton_pd(
            equity_value=500,
            equity_volatility=0.30,
            total_debt=400,
            risk_free_rate=0.04,
            time_horizon=0,
        )
    
    def test_merton_returns_valid_credit_metrics(self):
        result = calculate_merton_pd(
            equity_value=500,
            equity_volatility=0.30,
            total_debt=400,
            risk_free_rate=0.04,
        )

        assert result["asset_value"] > 0
        assert result["asset_volatility"] > 0
        assert result["distance_to_default"] > 0
        assert 0 <= result["probability_of_default_pct"] <= 100

    def test_merton_higher_debt_increases_default_risk(self):
        low_debt = calculate_merton_pd(
            equity_value=500,
            equity_volatility=0.30,
            total_debt=300,
            risk_free_rate=0.04,
        )

        high_debt = calculate_merton_pd(
            equity_value=500,
            equity_volatility=0.30,
            total_debt=700,
            risk_free_rate=0.04,
        )

        assert high_debt["probability_of_default_pct"] > low_debt[
            "probability_of_default_pct"
        ]

    def test_merton_invalid_equity_value(self):
        with pytest.raises(ValueError, match="equity_value must be positive"):
            calculate_merton_pd(
                equity_value=0,
                equity_volatility=0.30,
                total_debt=400,
                risk_free_rate=0.04,
            )

    def test_merton_invalid_debt(self):
        with pytest.raises(ValueError, match="total_debt must be positive"):
            calculate_merton_pd(
                equity_value=500,
                equity_volatility=0.30,
                total_debt=0,
                risk_free_rate=0.04,
            )


class TestMonteCarloSimulation:
    def test_simulation_invalid_ebitda_volatility(self):
        with pytest.raises(ValueError, match="ebitda_volatility must be non-negative"):
            run_cashflow_simulation(
            base_ebitda=100,
            ebitda_volatility=-0.10,
            annual_debt_service=20,
            max_leverage_covenant=4.0,
            min_coverage_covenant=3.0,
            net_debt=300,
        )


    def test_simulation_invalid_leverage_covenant(self):
        with pytest.raises(ValueError, match="max_leverage_covenant must be positive"):
            run_cashflow_simulation(
            base_ebitda=100,
            ebitda_volatility=0.20,
            annual_debt_service=20,
            max_leverage_covenant=0,
            min_coverage_covenant=3.0,
            net_debt=300,
        )


    def test_simulation_invalid_coverage_covenant(self):
        with pytest.raises(ValueError, match="min_coverage_covenant must be positive"):
            run_cashflow_simulation(
            base_ebitda=100,
            ebitda_volatility=0.20,
            annual_debt_service=20,
            max_leverage_covenant=4.0,
            min_coverage_covenant=0,
            net_debt=300,
        )    

    def test_simulation_is_reproducible_with_seed(self):
        kwargs = dict(
            base_ebitda=100,
            ebitda_volatility=0.20,
            annual_debt_service=20,
            max_leverage_covenant=4.0,
            min_coverage_covenant=3.0,
            net_debt=300,
            num_simulations=5000,
            random_seed=42,
        )

        first = run_cashflow_simulation(**kwargs)
        second = run_cashflow_simulation(**kwargs)

        assert first == second

    def test_simulation_returns_expected_statistics(self):
        result = run_cashflow_simulation(
            base_ebitda=100,
            ebitda_volatility=0.20,
            annual_debt_service=20,
            max_leverage_covenant=4.0,
            min_coverage_covenant=3.0,
            net_debt=300,
            num_simulations=5000,
            random_seed=42,
        )

        assert result["ebitda_mean"] > 0
        assert result["ebitda_median"] > 0
        assert result["ebitda_std"] > 0
        assert result["ebitda_p5"] < result["ebitda_p95"]

    def test_zero_volatility_produces_constant_ebitda(self):
        result = run_cashflow_simulation(
            base_ebitda=100,
            ebitda_volatility=0,
            annual_debt_service=20,
            max_leverage_covenant=4.0,
            min_coverage_covenant=3.0,
            net_debt=300,
            num_simulations=1000,
            random_seed=42,
        )

        assert result["ebitda_mean"] == pytest.approx(100)
        assert result["ebitda_median"] == pytest.approx(100)
        assert result["ebitda_std"] == pytest.approx(0)
        assert result["ebitda_p5"] == pytest.approx(100)
        assert result["ebitda_p95"] == pytest.approx(100)

    def test_covenant_breach_probability_is_valid_percentage(self):
        result = run_cashflow_simulation(
            base_ebitda=100,
            ebitda_volatility=0.20,
            annual_debt_service=20,
            max_leverage_covenant=4.0,
            min_coverage_covenant=3.0,
            net_debt=300,
            num_simulations=5000,
            random_seed=42,
        )

        assert 0 <= result["probability_of_breach_pct"] <= 100
        assert 0 <= result["leverage_breach_pct"] <= 100
        assert 0 <= result["coverage_breach_pct"] <= 100

    def test_tighter_covenants_do_not_reduce_breach_probability(self):
        loose = run_cashflow_simulation(
            base_ebitda=100,
            ebitda_volatility=0.20,
            annual_debt_service=20,
            max_leverage_covenant=5.0,
            min_coverage_covenant=2.0,
            net_debt=300,
            num_simulations=5000,
            random_seed=42,
        )

        tight = run_cashflow_simulation(
            base_ebitda=100,
            ebitda_volatility=0.20,
            annual_debt_service=20,
            max_leverage_covenant=3.0,
            min_coverage_covenant=4.0,
            net_debt=300,
            num_simulations=5000,
            random_seed=42,
        )

        assert tight["probability_of_breach_pct"] >= loose[
            "probability_of_breach_pct"
        ]

    def test_simulation_invalid_base_ebitda(self):
        with pytest.raises(ValueError, match="base_ebitda must be positive"):
            run_cashflow_simulation(
                base_ebitda=0,
                ebitda_volatility=0.20,
                annual_debt_service=20,
                max_leverage_covenant=4.0,
                min_coverage_covenant=3.0,
                net_debt=300,
            )

    def test_simulation_invalid_number_of_simulations(self):
        with pytest.raises(ValueError, match="num_simulations must be positive"):
            run_cashflow_simulation(
                base_ebitda=100,
                ebitda_volatility=0.20,
                annual_debt_service=20,
                max_leverage_covenant=4.0,
                min_coverage_covenant=3.0,
                net_debt=300,
                num_simulations=0,
            )
