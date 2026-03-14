#!/usr/bin/env python3
"""
End-to-end tests for the Hybrid Prediction System.

Exercises the full pipeline: feature completion → trend detection →
routing (ML model vs rule-based) → prediction output.
"""

import sys
import traceback
from hybrid_prediction_system import HybridPredictionSystem

REQUIRED_PRED_KEYS = {'Quarter', 'ARR', 'Pessimistic_ARR', 'Optimistic_ARR',
                      'YoY_Growth', 'YoY_Growth_Percent', 'QoQ_Growth_Percent'}

REQUIRED_META_KEYS = {'prediction_method', 'trend_analysis'}


def _assert_predictions_valid(predictions, metadata, label, expected_method=None):
    """Shared assertions for every test case."""
    errors = []

    # --- Structure ---
    if not isinstance(predictions, list) or len(predictions) != 4:
        errors.append(f"Expected 4 predictions, got {type(predictions).__name__} len={getattr(predictions, '__len__', lambda: '?')()}")
        return errors  # can't continue

    for i, pred in enumerate(predictions):
        missing = REQUIRED_PRED_KEYS - set(pred.keys())
        if missing:
            errors.append(f"Prediction {i} missing keys: {missing}")

        arr = pred.get('ARR')
        if arr is None or not isinstance(arr, (int, float)) or arr <= 0:
            errors.append(f"Prediction {i} ARR invalid: {arr}")

        pess = pred.get('Pessimistic_ARR', 0)
        opt = pred.get('Optimistic_ARR', 0)
        if pess > arr:
            errors.append(f"Prediction {i} Pessimistic ({pess:,.0f}) > ARR ({arr:,.0f})")
        if opt < arr:
            errors.append(f"Prediction {i} Optimistic ({opt:,.0f}) < ARR ({arr:,.0f})")

    # --- Metadata ---
    missing_meta = REQUIRED_META_KEYS - set(metadata.keys())
    if missing_meta:
        errors.append(f"Metadata missing keys: {missing_meta}")

    # --- Routing ---
    if expected_method:
        actual = metadata.get('prediction_method', '')
        if expected_method == 'ML':
            if 'ML' not in actual and 'ml' not in actual.lower():
                errors.append(f"Expected ML route, got '{actual}'")
        elif expected_method == 'Rule':
            if 'Rule' not in actual and 'rule' not in actual.lower() and 'Health' not in actual:
                errors.append(f"Expected Rule-based route, got '{actual}'")

    return errors


# =============================================================================
# 10 TEST CASES
# =============================================================================

def test_01_steady_growth_tier1(system):
    """Steady 20% annual growth, Tier 1 only → ML model."""
    tier1 = {
        'q1_arr': 10_000_000, 'q2_arr': 10_500_000,
        'q3_arr': 11_000_000, 'q4_arr': 12_000_000,
        'headcount': 80, 'sector': 'Data & Analytics'
    }
    preds, meta = system.predict_with_hybrid(tier1)
    errors = _assert_predictions_valid(preds, meta, "test_01", expected_method='ML')

    # Growth should be positive and reasonable
    for p in preds:
        if p['ARR'] < tier1['q4_arr']:
            errors.append(f"Predicted ARR ({p['ARR']:,.0f}) below current Q4 for a growing company")
    return errors


def test_02_steady_growth_tier2(system):
    """Steady growth + full Tier 2 metrics → ML model, health tier HIGH."""
    tier1 = {
        'q1_arr': 10_000_000, 'q2_arr': 10_500_000,
        'q3_arr': 11_000_000, 'q4_arr': 12_000_000,
        'headcount': 80, 'sector': 'Data & Analytics'
    }
    tier2 = {
        'gross_margin': 78, 'sales_marketing': 3_500_000,
        'cash_burn': -1_500_000, 'customers': 200,
        'churn_rate': 4, 'expansion_rate': 12
    }
    preds, meta = system.predict_with_hybrid(tier1, tier2)
    errors = _assert_predictions_valid(preds, meta, "test_02", expected_method='ML')

    ht = meta.get('health_tier')
    if ht and ht == 'LOW':
        errors.append(f"Healthy company got health_tier=LOW")
    return errors


def test_03_declining_company(system):
    """Consistent decline → Rule-based path, CONSISTENT_DECLINE trend."""
    tier1 = {
        'q1_arr': 5_000_000, 'q2_arr': 4_000_000,
        'q3_arr': 3_000_000, 'q4_arr': 2_000_000,
        'headcount': 40, 'sector': 'Infrastructure & Network'
    }
    preds, meta = system.predict_with_hybrid(tier1)
    errors = _assert_predictions_valid(preds, meta, "test_03", expected_method='Rule')

    trend = meta.get('trend_analysis', {}).get('trend_type', '')
    if 'DECLINE' not in trend and 'VOLATILE' not in trend:
        errors.append(f"Expected DECLINE or VOLATILE trend, got '{trend}'")
    return errors


def test_04_volatile_company(system):
    """Wild swings → Rule-based path, VOLATILE_IRREGULAR trend."""
    tier1 = {
        'q1_arr': 3_000_000, 'q2_arr': 1_500_000,
        'q3_arr': 4_000_000, 'q4_arr': 2_000_000,
        'headcount': 30, 'sector': 'Cyber Security'
    }
    preds, meta = system.predict_with_hybrid(tier1)
    errors = _assert_predictions_valid(preds, meta, "test_04", expected_method='Rule')

    trend = meta.get('trend_analysis', {}).get('trend_type', '')
    if 'VOLATILE' not in trend:
        errors.append(f"Expected VOLATILE trend, got '{trend}'")
    return errors


def test_05_flat_company(system):
    """Stagnant ARR → Rule-based path, FLAT_STAGNANT trend."""
    tier1 = {
        'q1_arr': 8_000_000, 'q2_arr': 8_100_000,
        'q3_arr': 7_900_000, 'q4_arr': 8_050_000,
        'headcount': 60, 'sector': 'Communication & Collaboration'
    }
    preds, meta = system.predict_with_hybrid(tier1)
    errors = _assert_predictions_valid(preds, meta, "test_05", expected_method='Rule')

    trend = meta.get('trend_analysis', {}).get('trend_type', '')
    if 'FLAT' not in trend:
        errors.append(f"Expected FLAT trend, got '{trend}'")
    return errors


def test_06_hyper_growth(system):
    """Triple-digit annual growth → ML model, predictions capped at 500%."""
    tier1 = {
        'q1_arr': 500_000, 'q2_arr': 800_000,
        'q3_arr': 1_400_000, 'q4_arr': 2_500_000,
        'headcount': 25, 'sector': 'Cyber Security'
    }
    preds, meta = system.predict_with_hybrid(tier1)
    errors = _assert_predictions_valid(preds, meta, "test_06", expected_method='ML')

    raw_preds = meta.get('raw_yoy_predictions', [])
    for v in raw_preds:
        if abs(v) > 500:
            errors.append(f"Raw prediction {v:.1f} exceeds ±500% cap")
    return errors


def test_07_low_health_override(system):
    """Growth trend looks normal but Tier 2 shows poor health → health metrics present."""
    tier1 = {
        'q1_arr': 6_000_000, 'q2_arr': 6_300_000,
        'q3_arr': 6_600_000, 'q4_arr': 7_000_000,
        'headcount': 90, 'sector': 'Marketing & Customer Experience'
    }
    tier2 = {
        'gross_margin': 35,
        'sales_marketing': 8_000_000,
        'cash_burn': -6_000_000,
        'customers': 50,
        'churn_rate': 20,
        'expansion_rate': 2
    }
    preds, meta = system.predict_with_hybrid(tier1, tier2)
    errors = _assert_predictions_valid(preds, meta, "test_07")

    # Health scoring should detect weakness even if override doesn't trigger
    ht = meta.get('health_tier')
    if ht is None:
        errors.append("Expected health_tier in metadata when Tier 2 provided")
    if ht == 'HIGH':
        errors.append(f"Distressed company should not get health_tier=HIGH")
    return errors


def test_08_trend_reversal(system):
    """Strong decline then recovery → TREND_REVERSAL → Rule-based."""
    tier1 = {
        'q1_arr': 4_000_000, 'q2_arr': 2_500_000,
        'q3_arr': 3_000_000, 'q4_arr': 4_500_000,
        'headcount': 45, 'sector': 'Data & Analytics'
    }
    preds, meta = system.predict_with_hybrid(tier1)
    errors = _assert_predictions_valid(preds, meta, "test_08", expected_method='Rule')

    trend = meta.get('trend_analysis', {}).get('trend_type', '')
    if 'REVERSAL' not in trend and 'VOLATILE' not in trend:
        errors.append(f"Expected REVERSAL or VOLATILE trend, got '{trend}'")
    return errors


def test_09_very_small_company(system):
    """$100K ARR micro-startup → should not crash."""
    tier1 = {
        'q1_arr': 80_000, 'q2_arr': 90_000,
        'q3_arr': 100_000, 'q4_arr': 120_000,
        'headcount': 5, 'sector': 'Other'
    }
    preds, meta = system.predict_with_hybrid(tier1)
    errors = _assert_predictions_valid(preds, meta, "test_09")

    for p in preds:
        if p['ARR'] <= 0:
            errors.append(f"Predicted non-positive ARR for small company: {p['ARR']}")
    return errors


def test_10_large_enterprise(system):
    """$100M ARR enterprise with 25% growth + Tier 2 → ML model, no crashes."""
    tier1 = {
        'q1_arr': 80_000_000, 'q2_arr': 87_000_000,
        'q3_arr': 93_000_000, 'q4_arr': 100_000_000,
        'headcount': 800, 'sector': 'Infrastructure & Network'
    }
    tier2 = {
        'gross_margin': 82, 'sales_marketing': 25_000_000,
        'cash_burn': -5_000_000, 'customers': 2000,
        'churn_rate': 3, 'expansion_rate': 15
    }
    preds, meta = system.predict_with_hybrid(tier1, tier2)
    errors = _assert_predictions_valid(preds, meta, "test_10", expected_method='ML')

    for p in preds:
        if p['ARR'] > tier1['q4_arr'] * 3:
            errors.append(f"Enterprise ARR prediction unreasonably high: {p['ARR']:,.0f}")
    return errors


# =============================================================================
# RUNNER
# =============================================================================

ALL_TESTS = [
    ("01 Steady growth (Tier 1)", test_01_steady_growth_tier1),
    ("02 Steady growth (Tier 2)", test_02_steady_growth_tier2),
    ("03 Declining company", test_03_declining_company),
    ("04 Volatile company", test_04_volatile_company),
    ("05 Flat/stagnant company", test_05_flat_company),
    ("06 Hyper-growth startup", test_06_hyper_growth),
    ("07 Low health override", test_07_low_health_override),
    ("08 Trend reversal", test_08_trend_reversal),
    ("09 Very small company", test_09_very_small_company),
    ("10 Large enterprise", test_10_large_enterprise),
]


def main():
    print("=" * 80)
    print("INITIALIZING HYBRID PREDICTION SYSTEM")
    print("=" * 80)
    system = HybridPredictionSystem()

    passed = 0
    failed = 0
    results = []

    for name, test_fn in ALL_TESTS:
        print(f"\n{'=' * 80}")
        print(f"RUNNING: {name}")
        print(f"{'=' * 80}")
        try:
            errors = test_fn(system)
            if errors:
                status = "FAIL"
                failed += 1
                detail = "; ".join(errors)
            else:
                status = "PASS"
                passed += 1
                detail = ""
        except Exception as e:
            status = "CRASH"
            failed += 1
            detail = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        results.append((name, status, detail))

    # --- Summary ---
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    for name, status, detail in results:
        icon = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{icon}] {name}")
        if detail:
            for line in detail.split("; "):
                print(f"         {line}")

    print(f"\n  {passed}/{passed + failed} tests passed")
    if failed:
        print(f"  {failed} FAILED")
        sys.exit(1)
    else:
        print("  All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
