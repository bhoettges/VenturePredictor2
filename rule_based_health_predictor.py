#!/usr/bin/env python3
"""
Rule-Based Health Assessment and Prediction System
===================================================

Uses research-backed metrics to assess company health and make transparent,
explainable ARR predictions based on health tier classification.

Based on industry benchmarks from McKinsey, BCG, and Bessemer Venture Partners.
"""

from typing import Dict, Optional, Tuple


class RuleBasedHealthPredictor:
    """
    Rule-based prediction system using research-backed health metrics.
    
    Health Metrics Assessed:
    1. ARR Growth Rate (YoY)
    2. Net Revenue Retention (NRR) / Churn Rate
    3. CAC Payback Period / LTV:CAC
    4. Rule of 40 (Growth + Profit Margin)
    5. Burn Rate and Runway
    """
    
    def __init__(self):
        """Initialize with industry benchmark thresholds."""
        
        # ARR Growth benchmarks (McKinsey: top quartile ~45%, median ~22%, bottom ~14%)
        self.HIGH_GROWTH_THRESHOLD = 0.40  # 40% YoY
        self.MODERATE_GROWTH_THRESHOLD = 0.15  # 15% YoY
        
        # NRR benchmarks (McKinsey: excellent ≥120%, good 100-120%, poor <100%)
        self.EXCELLENT_NRR_THRESHOLD = 120.0  # 120%
        self.GOOD_NRR_THRESHOLD = 100.0  # 100%
        
        # CAC Payback benchmarks (McKinsey: top quartile ~16 months, bottom ~47 months)
        self.EXCELLENT_CAC_PAYBACK_MONTHS = 18  # months
        self.GOOD_CAC_PAYBACK_MONTHS = 36  # months (3 years)
        
        # Rule of 40 threshold (BCG: should be ≥40%)
        self.RULE_OF_40_THRESHOLD = 40.0
        
        # Runway threshold (industry: minimum 12-18 months)
        self.MINIMUM_RUNWAY_MONTHS = 12
    
    def calculate_health_metrics(
        self,
        q1: float,
        q2: float,
        q3: float,
        q4: float,
        tier2_data: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate all health metrics from available data.
        
        Args:
            q1-q4: Historical ARR values
            tier2_data: Optional advanced metrics (gross_margin, sales_marketing, 
                       cash_burn, churn_rate, expansion_rate, customers, runway_months)
        
        Returns:
            Dict with all calculated health metrics and flags indicating which were estimated
        """
        metrics = {}
        metrics['_estimated_flags'] = {}  # Track which metrics were estimated
        
        # 1. ARR Growth Rate (YoY)
        arr_growth_yoy = ((q4 - q1) / q1) if q1 > 0 else 0
        metrics['arr_growth_yoy'] = arr_growth_yoy
        metrics['arr_growth_yoy_percent'] = arr_growth_yoy * 100
        
        # Calculate QoQ growth for recent momentum
        qoq1 = ((q2 - q1) / q1) if q1 > 0 else 0
        qoq2 = ((q3 - q2) / q2) if q2 > 0 else 0
        qoq3 = ((q4 - q3) / q3) if q3 > 0 else 0
        metrics['recent_momentum'] = qoq3  # Q3→Q4 most important
        
        # 2. Net Revenue Retention (NRR)
        if tier2_data and 'churn_rate' in tier2_data and 'expansion_rate' in tier2_data:
            churn_rate = tier2_data['churn_rate']
            expansion_rate = tier2_data['expansion_rate']
            # Convention: values are always in percentage points (e.g. 5 means 5%).
            # Guard against callers who pass decimals (e.g. 0.05 for 5%):
            # any value strictly below 1 is almost certainly a decimal encoding,
            # since sub-1% churn/expansion is unrealistic for SaaS.
            churn_pct = churn_rate if churn_rate >= 1 else churn_rate * 100
            expansion_pct = expansion_rate if expansion_rate >= 1 else expansion_rate * 100
            nrr = 100 - churn_pct + expansion_pct
            metrics['nrr'] = nrr
            metrics['churn_rate'] = churn_pct / 100
            metrics['expansion_rate'] = expansion_pct / 100
            metrics['_estimated_flags']['nrr'] = False
        else:
            # Estimate from ARR trend if no explicit churn/expansion
            # If declining, assume higher churn; if growing, assume expansion
            if arr_growth_yoy < 0:
                metrics['nrr'] = 95.0  # Estimated below 100%
                metrics['churn_rate'] = 0.08  # Estimated 8%
                metrics['expansion_rate'] = 0.03  # Estimated 3%
            else:
                metrics['nrr'] = 105.0  # Estimated above 100%
                metrics['churn_rate'] = 0.05  # Estimated 5%
                metrics['expansion_rate'] = 0.10  # Estimated 10%
            metrics['_estimated_flags']['nrr'] = True
        
        # 3. CAC Payback Period
        if tier2_data and 'sales_marketing' in tier2_data and 'customers' in tier2_data:
            sales_marketing = tier2_data['sales_marketing']
            customers = tier2_data['customers']
            
            if customers > 0 and sales_marketing > 0:
                cac = sales_marketing / customers  # CAC per customer
                arr_per_customer = q4 / customers
                # Payback = CAC / (ARR per customer / 12 months)
                if arr_per_customer > 0:
                    cac_payback_months = (cac / (arr_per_customer / 12))
                else:
                    cac_payback_months = 60  # Default high if can't calculate
                metrics['_estimated_flags']['cac_payback'] = False
            else:
                # Can't calculate, estimate
                if arr_growth_yoy > 0.30:
                    cac_payback_months = 20
                elif arr_growth_yoy > 0:
                    cac_payback_months = 30
                else:
                    cac_payback_months = 48
                metrics['_estimated_flags']['cac_payback'] = True
        else:
            # Estimate based on growth rate
            if arr_growth_yoy > 0.30:
                cac_payback_months = 20  # Fast growth = efficient acquisition
            elif arr_growth_yoy > 0:
                cac_payback_months = 30  # Moderate growth
            else:
                cac_payback_months = 48  # Declining = inefficient
            metrics['_estimated_flags']['cac_payback'] = True
        
        metrics['cac_payback_months'] = cac_payback_months
        
        # 4. Rule of 40
        if tier2_data and 'gross_margin' in tier2_data:
            gross_margin = tier2_data['gross_margin']
            # Estimate EBITDA margin (typically lower than gross margin)
            # For SaaS, EBITDA margin often = Gross Margin - (S&M + R&D + G&A as % of revenue)
            # Simplified: assume EBITDA margin ≈ Gross Margin - 30-40%
            ebitda_margin = gross_margin - 35  # Rough estimate
            metrics['_estimated_flags']['rule_of_40'] = False  # Gross margin provided, EBITDA estimated
        else:
            # Default assumptions
            gross_margin = 75
            ebitda_margin = -10  # Typical for growth-stage SaaS
            metrics['_estimated_flags']['rule_of_40'] = True
        
        rule_of_40 = (arr_growth_yoy * 100) + ebitda_margin
        metrics['rule_of_40'] = rule_of_40
        metrics['gross_margin'] = gross_margin
        metrics['ebitda_margin'] = ebitda_margin
        
        # 5. Burn Rate and Runway
        if tier2_data:
            cash_burn = abs(tier2_data.get('cash_burn', 0))  # Usually negative, take abs
            runway_months = tier2_data.get('runway_months', None)
            
            if 'cash_burn' in tier2_data:
                metrics['_estimated_flags']['cash_burn'] = False
            else:
                metrics['_estimated_flags']['cash_burn'] = True
            
            if runway_months is not None:
                metrics['_estimated_flags']['runway'] = False
            else:
                # Estimate runway if not provided
                # Rough estimate based on growth and ARR
                if arr_growth_yoy > 0.40:
                    runway_months = 15
                elif arr_growth_yoy > 0:
                    runway_months = 20
                else:
                    runway_months = 24
                metrics['_estimated_flags']['runway'] = True
        else:
            # Estimate from growth and ARR
            # High growth = higher burn typically
            if arr_growth_yoy > 0.40:
                cash_burn = q4 * 0.4  # 40% of ARR
                runway_months = 15
            elif arr_growth_yoy > 0:
                cash_burn = q4 * 0.25  # 25% of ARR
                runway_months = 20
            else:
                cash_burn = q4 * 0.15  # 15% of ARR (declining = cutting costs)
                runway_months = 24
            metrics['_estimated_flags']['cash_burn'] = True
            metrics['_estimated_flags']['runway'] = True
        
        metrics['cash_burn'] = cash_burn
        metrics['runway_months'] = runway_months
        
        return metrics
    
    def assess_health_tier(self, metrics: Dict) -> Tuple[str, Dict]:
        """
        Assess company health tier (High, Moderate, Low) based on metrics.
        
        Returns:
            Tuple of (health_tier, assessment_details)
        """
        assessment = {
            'tier': 'MODERATE',  # Default
            'score': 0,  # 0-100 score
            'strengths': [],
            'weaknesses': [],
            'benchmarks_met': [],
            'benchmarks_missed': []
        }
        
        score = 0
        max_score = 100
        
        # 1. ARR Growth (25 points)
        arr_growth = metrics['arr_growth_yoy_percent']
        if arr_growth >= self.HIGH_GROWTH_THRESHOLD * 100:
            score += 25
            assessment['strengths'].append(f"Strong ARR growth ({arr_growth:.1f}% YoY) - top quartile performance")
            assessment['benchmarks_met'].append("ARR Growth ≥40% (top quartile)")
        elif arr_growth >= self.MODERATE_GROWTH_THRESHOLD * 100:
            score += 15
            assessment['strengths'].append(f"Moderate ARR growth ({arr_growth:.1f}% YoY)")
        else:
            score += 5
            assessment['weaknesses'].append(f"Low ARR growth ({arr_growth:.1f}% YoY) - below industry median")
            assessment['benchmarks_missed'].append(f"ARR Growth <15% (below median ~22%)")
        
        # 2. NRR (25 points)
        nrr = metrics['nrr']
        if nrr >= self.EXCELLENT_NRR_THRESHOLD:
            score += 25
            assessment['strengths'].append(f"Excellent NRR ({nrr:.1f}%) - strong expansion exceeds churn")
            assessment['benchmarks_met'].append("NRR ≥120% (excellent)")
        elif nrr >= self.GOOD_NRR_THRESHOLD:
            score += 15
            assessment['strengths'].append(f"Good NRR ({nrr:.1f}%) - maintaining revenue base")
        else:
            score += 5
            assessment['weaknesses'].append(f"Poor NRR ({nrr:.1f}%) - losing net revenue (churn > expansion)")
            assessment['benchmarks_missed'].append("NRR <100% (losing net revenue)")
        
        # 3. CAC Payback (20 points)
        cac_payback = metrics['cac_payback_months']
        if cac_payback <= self.EXCELLENT_CAC_PAYBACK_MONTHS:
            score += 20
            assessment['strengths'].append(f"Efficient CAC payback ({cac_payback:.0f} months) - top quartile efficiency")
            assessment['benchmarks_met'].append(f"CAC Payback ≤{self.EXCELLENT_CAC_PAYBACK_MONTHS} months (top quartile)")
        elif cac_payback <= self.GOOD_CAC_PAYBACK_MONTHS:
            score += 12
            assessment['strengths'].append(f"Moderate CAC payback ({cac_payback:.0f} months)")
        else:
            score += 5
            assessment['weaknesses'].append(f"Slow CAC payback ({cac_payback:.0f} months) - inefficient acquisition")
            assessment['benchmarks_missed'].append(f"CAC Payback >{self.GOOD_CAC_PAYBACK_MONTHS} months (bottom quartile)")
        
        # 4. Rule of 40 (20 points)
        rule_of_40 = metrics['rule_of_40']
        if rule_of_40 >= self.RULE_OF_40_THRESHOLD:
            score += 20
            assessment['strengths'].append(f"Meets Rule of 40 ({rule_of_40:.1f}%) - balanced growth and profitability")
            assessment['benchmarks_met'].append("Rule of 40 ≥40%")
        elif rule_of_40 >= 30:
            score += 12
            assessment['strengths'].append(f"Near Rule of 40 ({rule_of_40:.1f}%)")
        else:
            score += 5
            assessment['weaknesses'].append(f"Below Rule of 40 ({rule_of_40:.1f}%) - growth/profitability imbalance")
            assessment['benchmarks_missed'].append("Rule of 40 <40%")
        
        # 5. Runway (10 points)
        runway = metrics['runway_months']
        if runway >= 18:
            score += 10
            assessment['strengths'].append(f"Strong runway ({runway:.0f} months) - sufficient capital")
        elif runway >= self.MINIMUM_RUNWAY_MONTHS:
            score += 6
            assessment['strengths'].append(f"Adequate runway ({runway:.0f} months)")
        else:
            score += 2
            assessment['weaknesses'].append(f"Short runway ({runway:.0f} months) - capital constraint risk")
            assessment['benchmarks_missed'].append(f"Runway <{self.MINIMUM_RUNWAY_MONTHS} months (minimum target)")
        
        # Determine tier based on score
        if score >= 75:
            assessment['tier'] = 'HIGH'
        elif score >= 50:
            assessment['tier'] = 'MODERATE'
        else:
            assessment['tier'] = 'LOW'
        
        assessment['score'] = score
        
        return assessment['tier'], assessment
    
    def predict_arr(
        self,
        q1: float,
        q2: float,
        q3: float,
        q4: float,
        sector: str,
        headcount: int,
        trend_analysis: Dict,
        tier2_data: Optional[Dict] = None
    ) -> Dict:
        """
        Predict ARR for next 4 quarters using rule-based health assessment.
        
        Args:
            q1-q4: Historical ARR values
            sector: Company sector
            headcount: Number of employees
            trend_analysis: Output from TrendDetector
            tier2_data: Optional advanced metrics
        
        Returns:
            Dict with predictions and detailed reasoning
        """
        
        # Step 1: Calculate health metrics
        metrics = self.calculate_health_metrics(q1, q2, q3, q4, tier2_data)
        
        # Step 2: Assess health tier
        health_tier, health_assessment = self.assess_health_tier(metrics)
        
        # Step 3: Apply prediction rules based on health tier
        import numpy as np
        trend_type = trend_analysis.get('trend_type', '')
        is_volatile = trend_type == 'VOLATILE_IRREGULAR'

        # For volatile companies, Q4 alone is unreliable as a base.
        # Use the median of all four quarters — robust to outlier quarters.
        if is_volatile:
            current_arr = float(np.median([q1, q2, q3, q4]))
        else:
            current_arr = q4

        arr_growth_yoy = metrics['arr_growth_yoy']
        recent_momentum = metrics['recent_momentum']
        
        # Determine projected growth rate based on health tier
        if health_tier == 'HIGH':
            # High health: Continue strong growth, but with deceleration as companies scale
            # Top quartile companies grow ~45%, but growth decays ~30% per year as they scale
            if arr_growth_yoy > 0.50:
                projected_annual_growth = 0.40  # Decelerate from 50%+ to 40%
            elif arr_growth_yoy > 0.40:
                projected_annual_growth = 0.35  # Decelerate from 40%+ to 35%
            else:
                projected_annual_growth = max(arr_growth_yoy * 0.85, 0.30)  # 15% decay, min 30%
            
            reasoning = f"High health company with strong fundamentals. Projecting {projected_annual_growth*100:.0f}% annual growth based on top-quartile benchmarks, with natural deceleration as company scales."
            confidence = "high"
            
        elif health_tier == 'MODERATE':
            # Moderate health: Modest growth, aligned with industry median (~20-22%)
            if arr_growth_yoy > 0:
                # If currently growing, project continued but moderated growth
                projected_annual_growth = max(arr_growth_yoy * 0.75, 0.15)  # 25% decay, min 15%
            else:
                # If declining, project stabilization or slight recovery
                projected_annual_growth = 0.10  # 10% recovery growth
            
            reasoning = f"Moderate health company. Projecting {projected_annual_growth*100:.0f}% annual growth, aligned with industry median performance (~22%)."
            confidence = "medium"
            
        else:  # LOW health
            # Low health: project based on severity of decline
            if arr_growth_yoy < 0:
                # Declining: use half the observed rate, floor at -25%
                projected_annual_growth = max(arr_growth_yoy * 0.5, -0.25)
                nrr_note = f" NRR is {metrics['nrr']:.0f}% (net revenue {'loss' if metrics['nrr'] < 100 else 'retention'})." if metrics.get('nrr') else ""
                reasoning = (
                    f"Low health company declining {arr_growth_yoy*100:.0f}% YoY.{nrr_note} "
                    f"Projecting {projected_annual_growth*100:.0f}% annual decline (half the observed rate, "
                    f"assuming some corrective action)."
                )
            elif metrics.get('nrr', 100) < 100:
                # Not declining yet but NRR < 100 signals future trouble
                projected_annual_growth = -0.05
                reasoning = f"Low health company with NRR <100% ({metrics['nrr']:.0f}%). Revenue base is eroding. Projecting -5% annual decline."
            else:
                projected_annual_growth = 0.05
                reasoning = f"Low health company with weak fundamentals. Projecting minimal 5% annual growth, acknowledging significant challenges."
            
            confidence = "medium"

        # For volatile patterns, override: the (Q4-Q1)/Q1 growth figure is
        # misleading because Q4 may be a trough or peak.  Use a conservative
        # neutral-to-moderate projection anchored to the median base ARR.
        if is_volatile:
            median_arr = float(np.median([q1, q2, q3, q4]))
            mean_arr = float(np.mean([q1, q2, q3, q4]))
            projected_annual_growth = 0.05  # conservative 5% for unpredictable companies
            reasoning = (
                f"Highly volatile ARR pattern (volatility {trend_analysis['metrics']['volatility']:.2f}). "
                f"Projecting from median ARR (${median_arr:,.0f}) with conservative 5% annual growth. "
                f"Wider confidence bands (+/-25%) reflect high uncertainty."
            )
            confidence = "low"

        # Convert annual growth to quarterly growth (approximate)
        # Annual growth = (1 + quarterly_growth)^4 - 1
        # So quarterly_growth ≈ annual_growth / 4 (simplified for small growth rates)
        quarterly_growth = projected_annual_growth / 4

        band = 0.25 if is_volatile else 0.10
        
        # Apply growth with slight deceleration over quarters
        predictions = []
        
        for i, quarter in enumerate(['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']):
            # Apply quarterly growth with slight deceleration
            growth_factor = quarterly_growth * (0.98 ** i)  # 2% deceleration per quarter
            next_arr = current_arr * (1 + growth_factor)
            
            # Calculate YoY growth (comparing to same quarter last year)
            base_arr = [q1, q2, q3, q4][i]
            yoy_growth = ((next_arr - base_arr) / base_arr) if base_arr > 0 else 0
            
            predictions.append({
                'Quarter': quarter,
                'ARR': next_arr,
                'Pessimistic_ARR': next_arr * (1 - band),
                'Optimistic_ARR': next_arr * (1 + band),
                'YoY_Growth': yoy_growth,
                'YoY_Growth_Percent': yoy_growth * 100,
                'QoQ_Growth_Percent': growth_factor * 100
            })
            
            current_arr = next_arr
        
        # Build detailed explanation
        key_assumption = f"Health tier: {health_tier} (Score: {health_assessment['score']}/100). "
        if health_assessment['strengths']:
            key_assumption += f"Strengths: {', '.join(health_assessment['strengths'][:2])}. "
        if health_assessment['weaknesses']:
            key_assumption += f"Concerns: {', '.join(health_assessment['weaknesses'][:2])}."
        
        # Build list of estimated metrics for transparency
        estimated_metrics_list = []
        if metrics.get('_estimated_flags', {}).get('nrr', False):
            estimated_metrics_list.append('NRR (churn/expansion not provided)')
        if metrics.get('_estimated_flags', {}).get('cac_payback', False):
            estimated_metrics_list.append('CAC Payback (S&M/customers not provided)')
        if metrics.get('_estimated_flags', {}).get('rule_of_40', False):
            estimated_metrics_list.append('Rule of 40 (gross margin not provided)')
        if metrics.get('_estimated_flags', {}).get('runway', False):
            estimated_metrics_list.append('Runway (not provided)')
        
        # Remove internal flags from metrics before returning
        metrics_clean = {k: v for k, v in metrics.items() if not k.startswith('_')}
        
        return {
            'success': True,
            'predictions': predictions,
            'health_tier': health_tier,
            'health_assessment': health_assessment,
            'health_metrics': metrics_clean,
            'estimated_metrics': estimated_metrics_list,
            'reasoning': reasoning,
            'confidence': confidence,
            'key_assumption': key_assumption,
            'prediction_method': 'Rule-Based Health Assessment'
        }
    
    def get_health_summary(self, health_assessment: Dict, metrics: Dict) -> str:
        """Generate human-readable health summary."""
        
        summary = f"""
Company Health Assessment: {health_assessment['tier']} HEALTH
Health Score: {health_assessment['score']}/100

Key Metrics:
  • ARR Growth (YoY): {metrics['arr_growth_yoy_percent']:.1f}%
  • Net Revenue Retention: {metrics['nrr']:.1f}%
  • CAC Payback Period: {metrics['cac_payback_months']:.0f} months
  • Rule of 40: {metrics['rule_of_40']:.1f}%
  • Cash Runway: {metrics['runway_months']:.0f} months

Strengths:
{chr(10).join('  ✓ ' + s for s in health_assessment['strengths'])}

Concerns:
{chr(10).join('  ⚠ ' + s for s in health_assessment['weaknesses'])}

Benchmarks Met:
{chr(10).join('  ✓ ' + b for b in health_assessment['benchmarks_met'])}

Benchmarks Missed:
{chr(10).join('  ✗ ' + b for b in health_assessment['benchmarks_missed'])}
"""
        return summary


def test_rule_based_predictor():
    """Test the rule-based predictor with different scenarios."""
    
    from trend_detector import TrendDetector
    
    detector = TrendDetector()
    predictor = RuleBasedHealthPredictor()
    
    test_cases = [
        {
            'name': 'HIGH HEALTH - Strong Growth Company',
            'q1': 1000000,
            'q2': 1400000,
            'q3': 2000000,
            'q4': 2800000,
            'sector': 'Data & Analytics',
            'headcount': 100,
            'tier2': {
                'gross_margin': 80,
                'sales_marketing': 1200000,
                'cash_burn': -800000,
                'churn_rate': 0.03,
                'expansion_rate': 0.25,
                'customers': 500,
                'runway_months': 24
            }
        },
        {
            'name': 'LOW HEALTH - Declining Company',
            'q1': 2000000,
            'q2': 1500000,
            'q3': 1000000,
            'q4': 500000,
            'sector': 'Data & Analytics',
            'headcount': 50,
            'tier2': {
                'gross_margin': 70,
                'sales_marketing': 200000,
                'cash_burn': -300000,
                'churn_rate': 0.15,
                'expansion_rate': 0.02,
                'customers': 100,
                'runway_months': 8
            }
        },
        {
            'name': 'MODERATE HEALTH - Steady Growth',
            'q1': 2000000,
            'q2': 2200000,
            'q3': 2400000,
            'q4': 2600000,
            'sector': 'Cyber Security',
            'headcount': 75,
            'tier2': None
        }
    ]
    
    for test in test_cases:
        print("\n" + "=" * 80)
        print(f"TEST: {test['name']}")
        print("=" * 80)
        
        trend_analysis = detector.detect_trend(test['q1'], test['q2'], test['q3'], test['q4'])
        
        result = predictor.predict_arr(
            q1=test['q1'],
            q2=test['q2'],
            q3=test['q3'],
            q4=test['q4'],
            sector=test['sector'],
            headcount=test['headcount'],
            trend_analysis=trend_analysis,
            tier2_data=test.get('tier2')
        )
        
        print(predictor.get_health_summary(result['health_assessment'], result['health_metrics']))
        
        print(f"\nPredictions:")
        for pred in result['predictions']:
            print(f"  {pred['Quarter']}: ${pred['ARR']:,.0f} "
                  f"(YoY: {pred['YoY_Growth_Percent']:+.1f}%, "
                  f"QoQ: {pred['QoQ_Growth_Percent']:+.1f}%)")
        
        print(f"\nReasoning: {result['reasoning']}")
        print(f"Confidence: {result['confidence']}")


if __name__ == "__main__":
    test_rule_based_predictor()

