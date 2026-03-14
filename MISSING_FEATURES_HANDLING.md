# Handling Missing Features in Rule-Based Health Predictor

## **Overview**

The rule-based health predictor is designed to work with **minimal required data** (Tier 1: Q1-Q4 ARR, sector, headcount) and gracefully handles missing Tier 2 features by using intelligent estimates based on ARR trends.

---

## **How It Works**

### **Tier 1 Data (Required)**
- ✅ Q1-Q4 ARR values
- ✅ Sector
- ✅ Headcount

**Always Calculated:**
- ARR Growth Rate (YoY) - directly from Q1-Q4 values
- Recent Momentum (Q3→Q4) - directly calculated

### **Tier 2 Data (Optional - Improves Accuracy)**

When Tier 2 data is **missing or incomplete**, the system uses **intelligent estimates** based on ARR trends:

| Metric | What's Needed | If Missing | Estimation Method |
|--------|--------------|------------|-------------------|
| **NRR** | `churn_rate`, `expansion_rate` | Estimated from ARR trend | Declining → NRR 95% (8% churn, 3% expansion)<br>Growing → NRR 105% (5% churn, 10% expansion) |
| **CAC Payback** | `sales_marketing`, `customers` | Estimated from growth rate | High growth (>30%) → 20 months<br>Moderate growth → 30 months<br>Declining → 48 months |
| **Rule of 40** | `gross_margin` | Estimated | Default: 75% gross margin, -10% EBITDA margin |
| **Runway** | `runway_months` | Estimated from growth | High growth → 15 months<br>Moderate → 20 months<br>Declining → 24 months |
| **Cash Burn** | `cash_burn` | Estimated from ARR | High growth → 40% of ARR<br>Moderate → 25% of ARR<br>Declining → 15% of ARR |

---

## **Transparency: Estimated vs. Provided Metrics**

The system **tracks and reports** which metrics were estimated:

### **Response Structure:**
```python
{
    'health_metrics': {
        'arr_growth_yoy_percent': -75.0,
        'nrr': 95.0,  # Estimated
        'cac_payback_months': 48,  # Estimated
        'rule_of_40': -85.0,  # Estimated
        'runway_months': 24  # Estimated
    },
    'estimated_metrics': [
        'NRR (churn/expansion not provided)',
        'CAC Payback (S&M/customers not provided)',
        'Rule of 40 (gross margin not provided)',
        'Runway (not provided)'
    ],
    'confidence': 'medium'  # Lower when using estimates
}
```

---

## **Example Scenarios**

### **Scenario 1: No Tier 2 Data**
**Input:**
- Q1-Q4 ARR: $2M → $500K (declining)
- No Tier 2 data provided

**Result:**
- ✅ System works with estimates
- ⚠️ All metrics estimated from ARR trends
- 📊 Confidence: Medium (estimates may reduce accuracy)
- 💡 Prediction: -5% annual decline

**Estimated Metrics:**
- NRR: 95% (estimated - company is declining)
- CAC Payback: 48 months (estimated - inefficient due to decline)
- Rule of 40: -85% (estimated - using default margins)
- Runway: 24 months (estimated - declining companies cut costs)

---

### **Scenario 2: Partial Tier 2 Data**
**Input:**
- Q1-Q4 ARR: $2M → $500K
- Tier 2: Only `gross_margin: 75` provided

**Result:**
- ✅ Rule of 40 calculated from provided gross margin
- ⚠️ Other metrics still estimated
- 📊 Confidence: Medium

**Estimated Metrics:**
- NRR: 95% (estimated - churn/expansion not provided)
- CAC Payback: 48 months (estimated - S&M/customers not provided)
- Runway: 24 months (estimated - not provided)

---

### **Scenario 3: Complete Tier 2 Data**
**Input:**
- Q1-Q4 ARR: $2M → $500K
- Tier 2: All metrics provided

**Result:**
- ✅ All metrics calculated from actual data
- ✅ No estimates used
- 📊 Confidence: High
- 💡 More accurate health assessment

**Calculated Metrics:**
- NRR: 87% (from actual churn 15%, expansion 2%)
- CAC Payback: 5 months (from actual S&M $200K, 100 customers)
- Rule of 40: -40% (from actual gross margin 70%)
- Runway: 8 months (from actual data)

---

## **Impact on Predictions**

### **Accuracy Impact:**

| Data Provided | Accuracy | Confidence |
|--------------|----------|------------|
| **Tier 1 Only** | Moderate | Medium |
| **Tier 1 + Partial Tier 2** | Good | Medium-High |
| **Tier 1 + Complete Tier 2** | Best | High |

### **Why Estimates Work:**

1. **ARR Growth is the strongest signal** - Most predictive metric
2. **Trend-based estimates** - Declining companies typically have:
   - Higher churn
   - Lower expansion
   - Longer CAC payback
   - Lower margins
3. **Conservative assumptions** - Estimates err on the side of caution

---

## **Best Practices**

### **For Users:**

1. **Minimum Viable:** Provide Tier 1 data (Q1-Q4 ARR) - system will work
2. **Better Accuracy:** Add Tier 2 metrics you have:
   - Churn rate + Expansion rate (for accurate NRR)
   - Sales & Marketing + Customers (for accurate CAC payback)
   - Gross margin (for accurate Rule of 40)
   - Runway (for accurate cash health)
3. **Best Accuracy:** Provide all Tier 2 metrics

### **For Developers:**

The system automatically:
- ✅ Detects missing metrics
- ✅ Applies appropriate estimates
- ✅ Reports which metrics were estimated
- ✅ Adjusts confidence levels
- ✅ Never fails due to missing data

---

## **Summary**

**The rule-based predictor:**
- ✅ **Works with minimal data** (Tier 1 only)
- ✅ **Gracefully handles missing features** (intelligent estimates)
- ✅ **Transparent about estimates** (reports what was estimated)
- ✅ **Improves with more data** (higher accuracy with Tier 2)
- ✅ **Never fails** (always produces predictions)

**Key Takeaway:** Users can start with just ARR data and get reasonable predictions. Providing additional Tier 2 metrics improves accuracy and confidence.


