# **Intelligent Forecasting: An AI-Powered Tool for Predicting Startup Revenue in Venture Capital DRAFT**

Balthasar Hoettges (807194)

Reichman University

January 2026

## **Executive Summary**

Venture capital (VC) investment decisions are made under extreme uncertainty. Early-stage companies often lack stable financial histories, and the available data is sparse, heterogeneous, and biased toward specific geographies and sectors. This project delivers a production-oriented forecasting system that predicts a startup’s next four quarters of ARR (Annual Recurring Revenue) using a LightGBM-based model, while explicitly addressing the core constraint of real-world usage: users cannot provide 100+ structured features at prediction time.

**The shipped system consists of:**

* **Tier-based input contract:** Users provide a minimal set of inputs (Tier 1\) and optional advanced metrics (Tier 2).  
* **Intelligent Feature Completion:** The system constructs a realistic full feature vector by identifying similar companies in the historical dataset and imputing missing metrics using robust statistics.  
* **Hybrid forecasting:** A routing layer detects non-standard trajectories (decline, stagnation, volatility) and falls back to a transparent Rule-Based Health Assessment that produces explainable projections using industry benchmarks.  
* **FastAPI service:** The system is exposed via production-ready API endpoints for single-company forecasts, batch CSV processing, chat-based explanation, and macroeconomic context. These endpoints are consumed by a front-end website, providing a user-friendly interface where investors and founders can input data and generate predictions in real-time.

On a held-out test set, the LightGBM model achieves strong performance with a final $R^2$ of around 80%. Performance varies by horizon (e.g., higher $R^2$ for the nearest quarter compared to the fourth quarter ahead), which is expected for multi-step forecasting in volatile startup environments. Predictions are returned with a pragmatic $\\pm 10\\%$ decision-support range to help VC users reason about risk and scenario bounds.

## **1\. Problem Statement and Success Criteria**

### **1.1 Prediction Task**

Given a company’s recent ARR trajectory (four quarters), headcount, and sector (plus optional Tier 2 metrics), the system forecasts:

1. **Q+1 to Q+4 ARR** (four quarters ahead).  
2. **Growth diagnostics** (QoQ and YoY rates) and uncertainty bounds.

Internally, the core model predicts YoY ARR growth rates for each of the next four quarters and converts those into absolute ARR by applying predicted YoY growth to last year’s corresponding quarter.

### **1.2 Real-World Constraint: Minimal User Input**

The training dataset contains many features per company-quarter, but a real VC workflow cannot assume that an investor or founder will provide that full feature set. The central engineering requirement is therefore: **Enable high-quality predictions from minimal inputs** while avoiding feature vectors that are out-of-distribution relative to the training data.

### **1.3 Success Criteria (VC \+ ML)**

* **Accuracy:** Strong predictive fit on held-out data ($R^2$/MAE), with horizon-wise reporting.  
* **Robustness under missingness:** Stable results with Tier 1 only; improved fidelity with Tier 2\.  
* **Behavior on edge cases:** Declining, flat, and volatile companies should not receive unrealistic “high-growth” forecasts.  
* **Explainability:** Routing decisions and rule-based projections must be auditable for investment decision support.  
* **Operational readiness:** API deployment, validation, error handling, and latency appropriate for interactive use.

## **2\. Data Overview**

### **2.1 Dataset Summary (Production Dataset)**

* **Observations:** 5,085 company-quarters  
* **Companies:** 354  
* **Raw Columns:** 122

This dataset is not *i.i.d.* (independent and identically distributed): it contains repeated measures per company and reflects venture-backed dynamics including high growth, extreme volatility, down periods, and reporting inconsistency.

### **2.2 Bias and Representativeness**

The dataset is geographically concentrated, with the US and a small set of other ecosystems dominating.

### **2.3 Missingness and Why It Matters**

Many features are sparse (often missing in 80–90%+ of rows). Naive "fill missing with 0" at inference creates unrealistic inputs and causes distribution shift, which motivated the development of **Intelligent Feature Completion**.

### 

### **2.4 Heavy Tails and Volatility**

ARR and growth metrics are heavy-tailed, spanning orders of magnitude. This requires:

* Logarithmic similarity for peer matching.  
* Robust feature imputation (weighted medians).  
* A hybrid path for non-standard trajectories.

## **3\. System Architecture (End-to-End)**

### **3.1 Production Dataflow**

The production API integrates several active modules:

1. **Request validation:** Managed via FastAPI and Pydantic.  
2. **Tier-based input parsing.**  
3. **Trend detection and routing.**  
4. **Forecasting Path:**  
   * **ML Route:** Intelligent feature completion $\\rightarrow$ LightGBM prediction.  
   * **Edge-case Route:** Rule-based health assessment projection.  
5. **Post-processing:** Calculation of QoQ, YoY, scenario bounds ($\\pm 10\\%$), and response formatting.

### **3.2 API Surface**

* POST /tier\_based\_forecast: Main forecast endpoint.  
* POST /predict\_csv: Batch forecasts from CSV upload.  
* POST /chat: Conversational explanation and analysis tools.  
* GET /makro-analysis: Macro indicators (VIX, MOVE, BVP, GPRH).  
* GET /health, GET /model\_info: System diagnostics.

## **4\. Tier-Based Input Contract**

### **4.1 Tier 1 (Required)**

Designed for the minimum information typically obtainable during diligence:

* **ARR:** Most recent four quarters (Q1–Q4).  
* **Headcount:** Current number of employees.  
* **Sector:** Constrained to a validated set of industry categories.

### **4.2 Tier 2 (Optional)**

Allows for increased forecast fidelity:

* Gross Margin & Cash Burn.  
* Sales & Marketing Spend.  
* Customer Count.  
* Churn and Expansion Rates.

## **5\. Intelligent Feature Completion (Core Contribution)**

### **5.1 The Necessity of Completion**

LightGBM expects a full engineered feature vector. Simple imputation harms performance; the system instead identifies "lookalike" companies to fill the gaps.

### **5.2 Similarity Matching**

The cohort of similar companies is found using:

* Logarithmic similarity on ARR (robust to scale).  
* Similarity on growth proxies and headcount.  
* Selection of the top 50 nearest companies as a peer set.

### **5.3 Robust Inference via Weighted Median**

Missing values are inferred from the peer set using a weighted median where weights equal the similarity score. This handles the heavy-tailed distributions typical in startup metrics.

### **5.4 Default Priors by ARR Scale**

If a feature is absent even in the peer set, the system falls back to size-dependent defaults (Small, Growth, or Large ARR regimes) to avoid implausible outputs like negative headcount.

## 

## 

## **6\. Hybrid Forecasting: ML Route \+ Rule-Based Health Assessment**

### **6.1 Rationale for Hybridization**

Pure ML systems trained on growth regimes can struggle with:

* Consistent decline or trend reversals.  
* Stagnant revenue or high volatility.

### **6.2 Trend Detection and Routing (6-Factor Analysis)**

A routing layer analyzes the last four quarters based on:

1. Overall $Q\_1 \\rightarrow Q\_4$ growth.  
2. QoQ sequence and momentum ($Q\_3 \\rightarrow Q\_4$).  
3. Consistency and volatility.  
4. Acceleration/deceleration.

### **6.3 Rule-Based Health Assessment (Edge-Case Method)**

For edge cases, a deterministic, benchmark-driven method is used:

* **Metrics:** Calculates NRR, CAC Payback, Rule of 40, and Runway.  
* **Tiers:** Assigns a health tier (High, Moderate, Low).  
* **Projection:** Uses conservative growth/decline rules for an auditable forecast.

## 

## 

## **7\. Model: LightGBM Multi-Output Forecasting**

### **7.1 Model Choice**

After benchmarking multiple model families, **LightGBM (GBDT)** was selected as the final approach because it best matches the realities of this codebase: **messy tabular startup data, high missingness at inference time, interaction-heavy SaaS dynamics, modest sample size, and production constraints (FastAPI, low latency, CPU-only)**.

#### **Why LightGBM was the right final choice:**

* **This is classic GBDT territory (tabular \+ heterogeneous \+ noisy).** The dataset is structured (not images/audio/text) and combines financial scalars (ARR, burn, S\&M), derived efficiency ratios (Magic Number, Burn Multiple), categorical descriptors (sector, geography, customer type), and time-derived features (lags/rolling windows) across repeated company-quarter observations. In this regime, **linear models tend to underfit**, while deep learning typically requires significantly more data and heavier regularisation to outperform GBDTs.  
* **Robustness under extreme missingness (even after feature completion).** A key system constraint is that end users cannot provide 150+ inputs. Missing fields are handled via an **Intelligent Feature Completion System** (peer similarity \+ weighted medians) to approximate a training-shaped feature vector. Even with completion, inference-time values remain imperfect. **Tree boosting remains stable under imputation noise, scaling differences, and monotonic distortions**, and does not require strict normalisation.  
* **The signal is non-linear and interaction-heavy (SaaS metrics).** Many drivers are conditional (e.g., burn is acceptable under strong growth/retention; S\&M spend only helps if it converts). LightGBM captures these interactions implicitly through a split structure without manual interaction engineering.  
* **Strong performance with limited data.** With \~5k company-quarter observations across \~350 companies, LightGBM provides a strong bias/variance tradeoff. Neural nets are more prone to overfitting and less stable under temporal splits at this scale.  
* **Production-friendly inference.** GBDT inference is fast and deterministic on CPU, with low operational risk and minimal dependency burden—well-suited to deployment behind an API.  
* **Multi-horizon forecasting support.** The model predicts **four horizons (Q1–Q4 ahead)**. Performance varies by horizon, with near-term quarters achieving higher R2R^2R2 than farther quarters—expected behaviour in multi-step forecasting for volatile startup environments.

### **7.2 Model architecture (training \+ inference design)**

* **Base learner:** Gradient-boosted decision trees via LightGBMRegressor  
* **Objective:** regression\_l1 (MAE-style loss) to improve robustness to outliers common in startup metrics  
* **Multi-horizon setup:** A **multi-output regression** design where the model produces **four forward-quarter predictions** (Q1–Q4 ahead), with per-horizon metrics tracked.  
* **Regularisation \+ generalisation controls:** Subsampling and feature column sampling to reduce variance and improve stability under distribution shift.  
* **Training stability:** High estimator cap with stopping based on validation performance (early stopping), rather than fixing the effective number of boosting rounds upfront.

### **Hyperparameter tuning (Optuna \+ CV \+ early stopping)**

Hyperparameters were tuned using a defensible and leakage-aware approach:

* **Optuna** for efficient hyperparameter search across the LightGBM space  
* **Cross-validation (CV)** to reduce sensitivity to any single split and improve robustness  
* **Early stopping** to prevent overfitting and select the optimal effective number of boosting rounds during training

### **Held-out performance and decision-support outputs**

On a held-out test set, the model achieves strong performance with a final **R2≈0.80R^2 \\approx 0.80R2≈0.80**. Performance degrades with longer horizons (near-term quarters outperform Q4-ahead), which aligns with expected uncertainty growth in multi-step forecasting. Predictions are returned with a pragmatic **±10%\\pm 10\\%±10%** decision-support range to help VC users reason about risk and scenario bounds.

### **7.2 Forecast Horizons and Evaluation**

![Figure 6. R² by forecast horizon.][image1]

### 

### **7.3 Feature Importance**

The top features provide interpretability, confirming that ARR growth and efficiency metrics are primary drivers in SaaS forecasting.

## **8\. Uncertainty Quantification**

All forecasts return three scenarios:

1. **Pessimistic:** \-10\\% of projection.  
2. **Realistic:** Core model projection.  
3. **Optimistic:** \+10\\% of projection.

This pragmatic band provides a risk-aware range for VC decision-making and a consistent format for stakeholder communication.

## **9\. Macroeconomic Context Module**

### **9.1 Macro layer: contextualising forecasts under changing market regimes**

The forecasting core is a company-level model, but VC outcomes depend heavily on the **macro regime**: fundraising timelines, valuation multiples, exit windows, and cost of capital shift meaningfully between **risk-on vs. risk-off** markets. As a result, identical ARR paths can imply very different VC outcomes depending on market conditions.

The macro module is included for a specific purpose:

* **Not to improve LightGBM accuracy directly** (macro indicators are **not** fed into the model),  
* **But to contextualize forecasts**: *given an ARR trajectory, what does the current regime imply for risk, fundraising, exits, and valuation sensitivity?*

This is academically defensible because it is a **decision-support overlay**, not a causal macro-finance model.

### **9\. 2 Indicators and data sources**

Each indicator is summarized over the last 12 months and translated into a traffic-light signal (green/amber/red) as a lightweight regime classifier for communication:

* **GPRH (Geopolitical Risk, Iacoviello):** monthly dataset, cached monthly *(gpr\_analysis.py)*. Higher risk maps to tighter liquidity and weaker M\&A appetite.  
* **VIX (Equity volatility, FRED VIXCLS):** cached weekly and resampled to monthly means (*vix\_analysis.py*). Captures risk sentiment affecting public comps, IPO windows, and follow-on appetite.  
* **MOVE (Rates volatility, Yahoo Finance ^MOVE):** cached weekly and resampled to monthly means (*move\_analysis.py*). Reflects discount-rate uncertainty and valuation compression/expansion.  
* **BVP / SaaS valuation proxy (FRED NASDAQEMCLOUDN):** public software benchmark used as a comps anchor (*bvp\_analysis.py*). Signals expansion vs compression in software valuations.

### **9.3 Design choice and limitation**

Macro variables are kept separate from the ML model to avoid **frequency mismatch (daily/monthly vs quarterly), alignment/leakage risks, faster regime shifts than reporting cycles, and reduced interpretability**. The module provides **context, not causality**: indicators are not LightGBM features, and regime thresholds are explicitly **heuristic segmentation** designed for decision support.

## **10\. Limitations and Threats to Validity**

### **10.1 Technical Bottlenecks (The ML Professor’s Critique)**

While the system achieves a respectable $R^2 \\approx 0.80$, several technical constraints limit its academic and predictive ceiling:

* **Data Sparsity and Imputation Bias:** The "Intelligent Feature Completion" system, while creative, introduces a risk of **feature circularity**. By imputing missing Tier 2 metrics based on peer similarity, the model may inadvertently reinforce existing correlations rather than discovering novel signals. If the peer set is biased, the imputed vector may drift from the ground truth, leading to "hallucinated" stability.  
* **Interpretability of the Conversational Layer:** The /chat endpoint represents a significant leap in usability, but it faces the classic **grounding problem**. While it can explain the "why" behind a forecast, it relies on the LLM’s ability to interpret LightGBM’s feature importance. Without strict **Retrieval-Augmented Generation (RAG)** tied to the underlying training data distributions, the bot may generate logically sound but factually unanchored narratives for edge-case startups.  
* **Temporal Leakage and Regime Shifts:** The startup ecosystem is prone to **structural breaks** (e.g., the post-2023 shift from "growth at all costs" to "capital efficiency"). A model trained primarily on historical quarters may overfit to signals that no longer hold in the current interest rate environment.  
* **Heuristic Uncertainty:** The current $\\pm 10\\%$ uncertainty band is a pragmatic product choice rather than a statistical one. To reach academic rigor, the system should transition to **Conformal Prediction** to provide mathematically guaranteed coverage intervals.

### **10.2 Operational and Commercial Risks (The Principal’s Assessment)**

From a venture perspective, the tool is a potent decision-support asset, but several "moat" and execution risks persist:

* **Data Moat and Replicability:** The model’s efficacy is tethered to its specific dataset of \~5,000 quarters. While the hybrid architecture is sophisticated, the "secret sauce" in AI-driven VC is proprietary data access. Without a flywheel to ingest non-public performance data at scale, this tool remains vulnerable to Tier-1 firms building internal versions with larger private datasets.  
* **Adoption Friction vs. Chat Interface:** The AI chatbot is the system's strongest "anti-friction" component. In the **"Age of AI,"** investment associates expect to query data rather than build dashboards. However, if the bot cannot link its reasoning back to specific peer-group benchmarks, it risks being dismissed as "AI fluff" during high-stakes investment committees.  
* **Cost-to-Serve and Scalability:** While the LightGBM model is lightweight, maintaining the **Macro Context Module** and the LLM-powered chat interface introduces recurring API costs and latency. For a VC firm, the "So What?" is whether this tool reduces the "Time-to-IC" (Investment Committee) enough to justify its operational overhead.

## 

## **11\. Conclusion: The New Standard for Quantitative VC**

This project delivers a production-ready ARR forecasting system that prioritizes robustness and human-centric design. By rejecting the "garbage in, garbage out" trap of naive imputation and instead opting for an **Intelligent Feature Completion** engine, the tool addresses the messy reality of early-stage financial data.

The defining characteristic of this system is its **AI-native accessibility**. By embedding a conversational layer, the tool transforms a complex gradient-boosting model into a collaborative partner. In an era where data density is overwhelming, the ability to ask, *"Why is the pessimistic scenario so aggressive for this SaaS company?"* and receive a context-aware response based on burn rates and macro volatility is not just a feature—it is a competitive necessity.

### **11.1 Strategic Next Steps**

To evolve this from a functional prototype to a market-leading investment engine, the following "missing links" should be prioritized:

1. **Agentic Data Ingestion:** Expand the chat interface into an **"Agentic Analyst"** that can ingest PDF pitch decks via OCR and automatically populate the Tier 1 and Tier 2 input vectors.  
2. **Probabilistic Guardrails:** Replace the $\\pm 10\\%$ heuristic with a Quantile LightGBM objective to produce true, mathematically derived probability distributions for revenue outcomes.  
3. **Valuation Sensitivity Overlay:** Integrate the Macro Module directly into the chat reasoning, allowing users to simulate how a "Risk-Off" market shift (spiking VIX/MOVE) would compress the valuation multiples of the predicted ARR path.

Ultimately, this system moves the needle from "gut-feel" investing toward a **quant-augmented approach**. It provides the analytical scaffolding necessary for VCs to navigate the inherent volatility of the startup ecosystem with both technical precision and commercial clarity.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAioAAAErCAIAAAB/5j/cAAAnGUlEQVR4Xu2djZOVZfnHf3/N7gqoDPgKlmI/E1NLdHxDMTXL0ZLKxpdSkuxlUIwx/TXWZC+Smak7VGaaOuXUBJqQOqHme0ImIY4pKgjC/m731ruH73Xuiz177XKWh89nPsPc536es9/D9/Cca8/uuv7PEMCO9A3T3PnFL34hOzUuGUZ3P8B+kJzVRE6osWjRoq7O38Vcd911p59+uu667FhD33nnnadnjDNvv/22fe5sw3anI7NmzRocHNTdAN///vdfeeWVIVNUeTwPPfRQ/zA//elP82nNu8ME5H90A/Z4mpd0bafGKMZP80UqvWSvXLmycbyK/VATitGNH93atXR87uyjsju7hpIrD+Cmm26y/1zTnw8//PA111zT3IeJBuMHlHT1nnDCCXfccUdzp3mFf+9735s8efLq1avLzr/+9a+pU6cuWbKk+RK2du3aGTNmnHLKKeU0+8ol4ydx5JFH5kX6HHbKlCnf+ta3yqF08tVXX33ooYfmx5N48MEH0/4///nPAw88MD3m5pmvvvrqPvvs8+STT+bQ/fbb76qrrkqL2267LT34xx57rJycHva+++578sknN+/+2muvpQf/2c9+tmwmUsQBBxyQPnLZSZ/jz549u3HK++Txc/3110+fPv3555/Pm82//syZM3//+9+Xm0Odysmkh5cecPqz7KQzFyxYcPjhh+ebV155ZSoqvUMtJwx1eqhnnHFG+hv97W9/Kztnn312enil/1ypTE37qJo7V1xxRXpsr7/+evPQtGnT0kMq737yhy3kMx955JEUfeaZZ37wkd6/7xFHHPGJT3yibBbmzJnz9NNP53XzAcjO448/3jxqz4QJBeMHlHzRlkv35ptvTq8j5WZazJ07Ny/mzZuXFn/5y1/y0VWrVqVFHj/p5TJvpsnUvG9eFPp2HD/Lli372c9+lhb9/f1nnXVWWtxwww3Nu6fx88ADD+R13rz00kvz+qWXXmqeOWnSpPzi2zdMXqRXxjSQtm/f3jzz/vvvT4vbb7+9uZmnZnrVa26miLz497//nRfpz3feeScttm3blk/LpPGTNu+999582l133VXOzzTXtZ28ef7556fF/Pnzm4/kxhtvLB8zPUFpcfzxx6chWk6wD3XLli3pQfYND+a8k56atEifN6Q5NNT9u5+0yEMxLVJ7eZHYuHHjU089Zb/4lj6x+NjHPpYWRx111MDAQFr89a9/bX60NETT4kc/+pETKutE+rwh71x77bXp79I8lPbLaIQJCOMHlHwxN18X7E05lF7Q80560ckvYc3T0sz4zGc+I5uZPkPZb56TX9xlUxaJW2+9NX0yLptlnQZb8175xTe9UNoz7d3TK2b5Ss6LL7743HPPpZt5DCeeffbZ5l2GhsdPGnXlZj6aXqyPPvrotPjPf/6z1157laPlnCZppzkm8wmbNm3Ki7yTHoacMNTpoaZFGrp555hjjvnud7+bTy7PWqY2fixDwx8wL8pp+c8f/vCHeUfGT3ozXc5v3vHCCy889dRTZbO5tjv/fRzDlGfBctlll11wwQW6CxMGxg8o+VL/+Mc/ft9995Wb+c9HH300L5pnNnfSACjjR5AzM33mi2+J9Om53PfOO+/MJ5dzyrq5WW52PDMFpdfEsrl27dq0eOGFFxo5HT5m+YAbNmwom4n0VqN5x+ZdhobHz3e+851ysxzNi/TebvPmzeVo81CT9P5mn332KTdnzpyZ3+iUMxcuXLjjQ+j8UBNf+9rX8glp7KXHlnYee+yxvJPfiAzVx0/Hna9+9avnnHOObKY/16xZk3ea4+ftt99ufpzm+vnnny/37XiC3ZF1+idXbgo///nPDzroIN2FCQPjB5RyeafF0qVLf/vb35bNjRs32heC5s43vvEN++6nYDf7Oo2fvK9b5nXHbvqvZR3HT8cz7WYaGPmLfpn0luK4447L30zqSHqJ/9znPlduykfe6d8u88wzz8gjyT+XUTZvuukm+/JqH+qCBQsOOOCAfPOQQw7J46dQvnXf1fhJg3D69OmyWVod2nH89O34Tqv5MdPwyF9zs503cY7akwvpM4DmdwRhosH4AaVcz33D2M0tW7YMDf9oQN784he/ePDBB5ejZfx8/etfL5vLli1rfpBCX338vPXWW2UtC9m8/PLLy/qee+6pnWnHz6ZNm8rR2267zQlqfpnrQx/60OLFi5tfGZs6dWr+Nlghf+8nrz/60Y+edNJJeZ3ePqa3laeddtp/T/2AZmhzs7xwOw8vsXXr1tpDTW+bzj333HL+okWLmncs6/SGJj2VZbN5qONOWaS/zt5775137PhJm3/84x/zZibt/OpXvyrrl19+OS+aJ5S13ZGjqU97fiY9L8uXL9ddmDAwfkApF/Ps2bPta01+2c3knUQaP3nnlltuKZ9BT5o0KW+eccYZead5l7LTcfwMDR/KPPzww2WnebSsS9DChQvt0bK24yct0qfe+b7pUDmz493Tq3M+87DDDss7v/vd7/JOmb6FNH7yD18kDjzwwOah5gdv0nE//1xDxs7jxLp168oJ6eS8aR9qOeeXv/xlfrR//vOfy+a777479EFW/uZZoc88qrJTost3ufrM+Bkc/qGVJuXMTP5yYt7JC1mXnfLDHR2P5i/P2n3dgokE4wdgl8Jr4ihIA7L8XN8ISZ8nyQ/CwUSD8QOwizj++OP7PvhaE3RLeo+rWy6M+YnPKMdPx6c2vRlPn25cfPHFegAAAGBHRjN+8pfLZbN8F3fFihX2KAAAQJOux08aLfk/bpf9/v7+rVu35rU9CgAA0KTr8ZOxA6a5Y48CAAA0Ga/xU94JdWQlAADs2YzX+Gkc6cDKkf1SfQAAaCVjOX7mzp27atWqvLZHBcYPAMCezBiMnwcffLCs+4Z/n8fAwMBFF13031M7wfgBANiTGf34CcL4AQDYk2H8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8AABAD2D8wM6Zcd0MHLlaHwB0gvEDO8e+wqKj1gcAnWD8wM6xr7DoqPUBQCcYP7Bz7CssOmp9ANAJxg/sHPsKi45aHwB0gvEDO8e+wqKj1gcAnWD8wM6xr7DoqPUBQCcYP7Bz7CssOmp9ANAJxg/sHPsKi45aHwB0gvEDO8e+wqKj1gcAnWD8wM6xr7DoqPUBQCcYP7Bz7CssOmp9ANCJ0YyfdevWTZ069eKLL9YDQ0OPP/54OnTeeefpAQPjZzfCvsKio9YHAJ3oevxs2rSpr68vLVasWJEXhdWrV0+ZMiUtnn76aTlkYfzsRthXWHTU+gCgE12Pn/7+/q1bt+a1zJizzz77jjvu6HjIwvjZjbCvsOio9QFAJ7oeP825YmdM3wc899xzckhg/OxG2FdYdNT6AKAT0fFT3gklnnjiiRkz3rv21q9fbyeTsHJkpDMP/b9DsStHXu9OWbVq1RDjp0tTY6k3rRIADKHx0ziyw8299trrzTffbBxUVo743Y+9vNFXGwxjI9BR6wOATnQ9fubOnZs/Ix5yx89I3v3oVgV7eaOvNhjGRqCj1gcAneh6/AwNj5ZFixYNDAxcdNFF6ebg4GAeNq+++mo+tP/++0+ePFnvtiOMn/FTGwxjI9BR6wOAToxm/IwJjJ/xUxsMYyPQUesDgE4wflqoNhjGRqCj1gcAnWD8tFBtMIyNQEetDwA6wfhpodpgGBuBjlofAHSC8dNCtcEwNgIdtT4A6ATjp4Vqg2FsBDpqfQDQCcZPC9UGw9gIdNT6AKATjJ8Wqg2GsRHoqPUBQCcYPy1UGwxjI9BR6wOATjB+Wqg2GMZGoKPWBwCdYPy0UG0wjI1AR60PADrB+Gmh2mAYG4GOWh8AdILx00K1wTA2Ah21PgDoBOOnhWqDYWwEOmp9ANAJxk8L1QbD2Ah01PoAoBOMnxaqDYaxEeio9QFAJxg/LVQbDGMj0FHrA4BOMH5aqDYYxkago9YHAJ1g/LRQbTCMjUBHrQ8AOsH4aaHaYBgbgY5aX5hD/+9Q7EptECYkjJ8Wqg2GsRHoqPWFsRHoqw3ChITx00K1wTA2Ah21vjA2An21QZiQMH5aqDYYxkago9YXxkagrzYIExLGTwvVBsPYCHTU+sLYCPTVBmFCwvhpodpgGBuBjlpfGBuBvtogTEgYPy1UGwxjI9BR6wtjI9BXG4QJCeOnhWqDYWwEOmp9YWwE+mqDMCFh/LRQbTCMjUBHrS+MjUBfbRAmJIyfFqoNhrER6Kj1hbER6KsNwoSE8dNCtcEwNgIdtb4wNgJ9tUGYkDB+Wqg2GMZGoKPWF8ZGoK82CBMSxk8L1QbD2Ah01PrC2Aj01QZhQsL4aaHaYBgbgY5aXxgbgb7aYBgbgY5aXwXGTwvVBsPYCHTU+sLYCPTVBsPYCHTU+iowflqoNhjGRqCj1hfGRqCvNhjGRqCj1leB8dNCtcEwNgIdtb4wNgJ9tcEwNgIdtb4KjJ8Wqg2GsRHoqPWFsRHoqw2GsRHoqPVVYPy0UG0wjI1AR60vjI1AX20wjI1AR62vAuOnhWqDYWwEOmp9YWwE+mqDYWwEOmp9FRg/LVQbDGMj0FHrC2Mj0FcbDGMj0FHrq8D4aaHaYBgbgY5aXxgbgb7aYBgbgY5aXwXGTwvVBsPYCHTU+sLYCPTVBsPYCHTU+iowflqoNhjGRqCj1hfGRqCvNhjGRqCj1ldhNOOnr69v8eLFAwMDX/7yl+2hSy+9dNasWWkhhwTGz/ipDYaxEeio9YWxEeirDYaxEeio9VXoevzMmTNn+fLleS0zJh164okn8vroo49uHrIwfsZPbTCMjUBHrS+MjUBfbTCMjUBHra9C1+OnOXJk/OSbxxxzzHHHHdfc7wjjZ/zUBsPYCHTU+sLYCPTVBsPYCHTU+ipEx8+WLVuaNwcGBtLitddeG8kX30bCEE9894683p2yatUqnoJuTY2l3rTK0UL/o5BLoLeOvP/Q+Gkc8Q5ZUvC2ETDEE9+9qTTtMQBPQbfSf8/lKeitI+y/6/HT39+/ffv2vA6OH92qYP9u6KsNhrER6Kj1hbER6KsNhrER6Kj1Veh6/GzevDmPluXLl8uMeeGFF/LO+vXrGT89VBsMYyPQUesLYyPQVxsMYyPQUeur0PX4SWzYsGH69Onz58/PNwcHB8uw2bhx48EHH/zJT36ynFyD8TN+aoNhbAQ6an1hbAT6aoNhbAQ6an0VRjN+xgTGz/ipDYaxEeio9YWxEeirDYaxEeio9VVg/LRQbTCMjUBHrS+MjUBfbTCMjUBHra8C46eFaoNhbAQ6an1hbAT6aoNhbAQ6an0VGD8tVBsMYyPQUesLYyPQVxsMYyPQUeurwPhpodpgGBuBjlpfGBuBvtpgGBuBjlpfBcZPC9UGw9gIdNT6wtgI9NUGw9gIdNT6KjB+Wqg2GMZGoKPWF8ZGoK82GMZGoKPWV4Hx00K1wTA2Ah21vjA2An21wTA2Ah21vgqMnxaqDYaxEeio9YWxEeirDYaxEeio9VVg/LRQbTCMjUBHrS+MjUBfbTCMjUBHra8C46eFaoNhbAQ6an1hbAT6aoNhbAQ6an0VGD8tVBsMYyPQUesLYyPQVxsMYyPQUeurwPhpodpgGBuBjlpfGBuBvtpgGBuBjlpfBcZPC9UGw9gIdNT6wtgI9NUGw9gIdNT6KjB+Wqg2GMZGoKPWF8ZGoK82GMZGoKPWV+G98bN+/fr+YQYGBi688MJyLO00zhxjGD/jpzYYxkago9YXxkagrzYYxkago9ZX4b3x0xwzaX3ssceWddkfcxg/46c2GMZGoKPWF8ZGoK82GMZGoKPWV+G98ZPe9DS30tT5yle+khfN/bGF8TN+aoNhbAQ6an1hbAT6aoNhbAQ6an0V3n/3s3379uZuGkg/+clPGD+7qdpgGBuBjlpfGBuBvtpgGBuBjlpfhffGT5o9A8M0D6SbjJ/dVG0wjI1AR60vjI1AX20wjI1AR62vgveTb3//+991a+xg/Iyf2mAYG4GOWl8YG4G+2mAYG4GOWl+F98fPsmXLZs+evW7dunJg+fLlvPvZTdUGw9gIdNT6wtgI9NUGw9gIdNT6Krw3fmbMmJEmzYknnjgwMPDUU08NDX/lLfGnP/1JTx87GD/jpzYYxkago9YXxkagrzYYxkago9ZXQX/wOn/L57DDDmucMy4wfsZPbTCMjUBHrS+MjUBfbTCMjUBHra+C/uB1Wo/rt3wKjJ/xUxsMYyPQUesLYyPQVxsMYyPQUeur0GH8NI6OI4yf8VMbDGMj0FHrC2Mj0FcbDGMj0FHrq8D4aaHaYBgbgY5aXxgbgb7aYBgbgY5aX4X3x08h/+a3gp4+djB+xk9tMIyNQEetL4yNQF9tMIyNQEetr8J74+efdfT0sYPxM35qg2FsBDpqfWFsBPpqg2FsBDpqfRXe/+9+dj2Mn/FTGwxjI9BR6wtjI9BXGwxjI9BR66vA+Gmh2mAYG4GOWl8YG4G+2mAYG4GOWl8Fxk8L1QbD2Ah01PrC2Aj01QbD2Ah01PoqMH5aqDYYxkago9YXxkagrzYYxkago9ZXgfHTQrXBMDYCHbW+MDYCfbXBMDYCHbW+CoyfFqoNhrER6Kj1hbER6KsNhrER6Kj1VWD8tFBtMIyNQEetL4yNQF9tMIyNQEetrwLjp4Vqg2FsBDpqfWFsBPpqg2FsBDpqfRUYPy1UGwxjI9BR6wtjI9BXGwxjI9BR66vA+Gmh2mAYG4GOWl8YG4G+2mAYG4GOWl8Fxk8L1QbD2Ah01PrC2Aj01QbD2Ah01PoqMH5aqDYYxkago9YXxkagrzYYxkago9ZXgfHTQrXBMDYCHbW+MDYCfbXBMDYCHbW+CqMZP9OmTesbZsuWLXpsmHRItwyMn/FTGwxjI9BR6wtjI9BXGwxjI9BR66vQ9fhZunTpvHnz8rrjmJkxY0bHfYHxM35qg2FsBDpqfWFsBPpqg2FsBDpqfRW6Hj/N0WLHzD/+8Y9Pf/rTdt/C+Bk/tcEwNgIdtb4wNgJ9tcEwNgIdtb4KYzx+8o7dt6TgbSNgiCe+e1Np2mMAnoJupf+ey1PQW0fY/1iOn3JzhONnJAzxxHfvyOvdKatWreIp6NbUWOpNqxwt9D8KuQR668j7H8vx06R5yLKSL76Nm9pgGBuBjlpfGBuBvtpgGBuBjlpfha7Hz+bNm/NoWb58eW3G1PabMH7GT20wjI1AR60vjI1AX20wjI1AR62vQtfjJ7Fhw4bp06fPnz8/3xwcHJR5w/jprdpgGBuBjlpfGBuBvtpgGBuBjlpfhdGMnzGB8TN+aoNhbAQ6an1hbAT6aoNhbAQ6an0VGD8tVBsMYyPQUesLYyPQVxsMYyPQUeurwPhpodpgGBuBjlpfGBuBvtpgGBuBjlpfBcZPC9UGw9gIdNT6wtgI9NUGw9gIdNT6KjB+Wqg2GMZGoKPWF8ZGoK82GMZGoKPWV4Hx00K1wTA2Ah21vjA2An21wTA2Ah21vgqMnxaqDYaxEeio9YWxEeirDYaxEeio9VVg/LRQbTCMjUBHrS+MjUBfbTCMjUBHra8C46eFaoNhbAQ6an1hbAT6aoNhbAQ6an0VGD8tVBsMYyPQUesLYyPQVxsMYyPQUeurwPhpodpgGBuBjlpfGBuBvtpgGBuBjlpfBcZPC9UGw9gIdNT6wtgI9NUGw9gIdNT6KjB+Wqg2GMZGoKPWF8ZGoK82GMZGoKPWV4Hx00K1wTA2Ah21vjA2An21wTA2Ah21vgqMnxaqDYaxEeio9YWxEeirDYaxEeio9VVg/LRQbTCMjUBHrS+MjUBfbTCMjUBHra8C46eFaoNhbAQ6an1hbAT6aoNhbAQ6an0VGD8tVBsMYyPQUesLYyPQVxsMYyPQUeurwPhpodpgGBuBjlpfGBuBvtpgGBuBjlpfBcZPC9UGw9gIdNT6wtgI9NUGw9gIdNT6KjB+Wqg2GMZGoKPWF8ZGoK82GMZGoKPWV4Hx00K1wTA2Ah21vjA2An21wTA2Ah21vgqMnxaqDYaxEeio9YWxEeirDYaxEeio9VVg/LRQbTCMjUBHrS+MjUBfbTCMjUBHra8C46eFaoNhbAQ6an1hbAT6aoNhbAQ6an0VGD8tVBsMYyPQUesLYyPQVxsMYyPQUeurwPhpodpgGBuBjlpfGBuBvtpgGBuBjlpfBcZPC9UGw9gIdNT6wtgI9NUGw9gIdNT6KjB+Wqg2GMZGoKPWF8ZGoK82GMZGoKPWV4Hx00K1wTA2Ah21vjA2An21wTA2Ah21vgqMnxaqDYaxEeio9YWxEeirDYaxEeio9VVg/LRQbTCMjUBHrS+MjUBfbTCMjUBHra8C46eFaoNhbAQ6an1hbAT6aoNhbAQ6an0VGD8tVBsMYyPQUesLYyPQVxsMYyPQUeurwPhpodpgGBuBjlpfGBuBvtpgGBuBjlpfBcZPC9UGw9gIdNT6wtgI9NUGw9gIdNT6KjB+Wqg2GMZGoKPWF8ZGoK82GMZGoKPWV4Hx00K1wTA2Ah21vjA2An21wTA2Ah21vgqMnxaqDYaxEeio9YWxEeirDYaxEeio9VVg/LRQbTCMjUBHrS+MjUBfbTCMjUBHra8C46eFaoNhbAQ6an1hbAT6aoNhbAQ6an0VGD8tVBsMYyPQUesLYyPQVxsMYyPQUeurMJrxM23atL5htmzZIofyfmLFihVySGD8jJ/aYBgbgY5aXxgbgb7aYBgbgY5aX4Wux8/SpUvnzZuX12nMNA+lsfToo492PGRh/Iyf2mAYG4GOWl8YG4G+2mAYG4GOWl+FrsdPc67IjFm7dm1ZM356qDYYxkago9YXxkagrzYYxkago9ZXYSzHT2H9+vW1Q4UUvG0EDPHEd28qTXsMwFPQrfTfc3kKeusI+x/78bNmzZqO+8LKkTHEE9+9I693p6xatYqnoFtTY6k3rXK00P8o5BLorSPvfyzHz3XXXWc3O7KSL76Nm9pgGBuBjlpfGBuBvtpgGBuBjlpfha7Hzy233HLWWWfltUyau+++e4SzZ4jxM55qg2FsBDpqfWFsBPpqg2FsBDpqfRW6Hj+J/fbbr2+Y/IPXg4ODeerkzYLebUcYP+OnNhjGRqCj1hfGRqCvNhjGRqCj1ldhNONnTGD8jJ/aYBgbgY5aXxgbgb7aYBgbgY5aXwXGTwvVBsPYCHTU+sLYCPTVBsPYCHTU+iowflqoNhjGRqCj1hfGRqCvNhjGRqCj1leB8dNCtcEwNgIdtb4wNgJ9tcEwNgIdtb4KjJ8Wqg2GsRHoqPWFsRHoqw2GsRHoqPVVYPy0UG0wjI1AR60vjI1AX20wjI1AR62vAuOnhWqDYWwEOmp9YWwE+mqDYWwEOmp9FRg/LVQbDGMj0FHrC2Mj0FcbDGMj0FHrq8D4aaHaYBgbgY5aXxgbgb7aYBgbgY5aXwXGTwvVBsPYCHTU+sLYCPTVBsPYCHTU+iowflqoNhjGRqCj1hfGRqCvNhjGRqCj1leB8dNCtcEwNgIdtb4wNgJ9tcEwNgIdtb4KjJ8Wqg2GsRHoqPWFsRHoqw2GsRHoqPVVYPy0UG0wjI1AR60vjI1AX20wjI1AR62vAuOnhWqDYWwEOmp9YWwE+mqDYWwEOmp9FRg/LVQbDGMj0FHrC2Mj0FcbDGMj0FHrq8D4aaHaYBgbgY5aXxgbgb7aYBgbgY5aXwXGTwvVBsPYCHTU+sLYCPTVBsPYCHTU+iowflqoNhjGRqCj1hfGRqCvNhjGRqCj1leB8dNCtcEwNgIdtb4wNgJ9tcEwNgIdtb4KjJ8Wqg2GsRHoqPWFsRHoqw2GsRHoqPVVYPy0UG0wjI1AR60vjI1AX20wjI1AR62vAuOnhWqDYWwEOmp9YWwE+mqDYWwEOmp9FRg/LVQbDGMj0FHrC2Mj0FcbDGMj0FHrq8D4aaHaYBgbgY5aXxgbgb7aYBgbgY5aXwXGTwvVBsPYCHTU+sLYCPTVBsPYCHTU+iowflqoNhjGRqCj1hfGRqCvNhjGRqCj1leB8dNCtcEwNgIdtb4wNgJ9tcEwNgIdtb4KjJ8Wqg2GsRHoqPWFsRHoqw2GsRHoqPVVYPy0UG0wjI1AR60vjI1AX20wjI1AR62vAuOnhWqDYWwEOmp9YWwE+mqDYWwEOmp9FRg/LVQbDGMj0FHrC2Mj0FcbDGMj0FHrq8D4aaHaYBgbgY5aXxgbgb7aYBgbgY5aXwXGTwvVBsPYCHTU+sLYCPTVBsPYCHTU+iowflqoNhjGRqCj1hfGRqCvNhjGRqCj1leB8dNCtcEwNgIdtb4wNgJ9tcEwNgIdtb4KjJ8Wqg2GsRHoqPWFsRHoqw2GsRHoqPVVYPy0UG0wjI1AR60vjI1AX20wjI1AR62vAuOnhWqDYWwEOmp9YWwE+mqDYWwEOmp9FRg/LVQbDGMj0FHrC2Mj0FcbDGMj0FHrq8D4aaHaYBgbgY5aXxgbgb7aYBgbgY5aXwXGTwvVBsPYCHTU+sLYCPTVBsPYCHTU+iqMZvysW7du6tSpF198sR4YGnrhhRf23XffK664Qg8YGD/jpzYYxkago9YXxkagrzYYxkago9ZXoevxs2nTpr6+vrRYsWJFXhTWr1+fd+68886BgYHmIQvjZ/zUBsPYCHTU+sLYCPTVBsPYCHTU+ip0PX76+/u3bt2a1zJ+mjflkIXxM35qg2FsBDpqfWFsBPpqg2FsBDpqfRW6Hj/OjHEOWRg/46c2GMZGoKPWF8ZGoK82GMZGoKPWVyE6fso7IXuorDuyEgAA9mxC46dxxDsEAAAgdDd+5s6du2rVqryWGTNr1qyXX3654yEAAAChu/EzNDxaFi1aNDAwcNFFF6Wbg4ODZdikxZIlS9Kf11577Q73AQAA2JGuxw8AAEAcxg8AAPQAxg8AAPQAxg8AAPQAxg8AAPQAxg8AAPSAPWj89BlmzZqlJ3XJmWeeqVs7Mnny5BInh8rvb91z+G/1H6BndI//QTZs2FCy7r///rJ/yCGH5M1zzjmncXrLKVUU4pfAm2++qVs7UrsEOm62nvK3HsO/vv9BmpfA0UcfLUf7+/sffPBB2dxl7EHjp+A/W13hf6h0dOPGjc2bZf3AAw+M1T++3Y70zz3+qldwOly3bl3z6KRJk66++uq0+PWvf33sscfmzXTCO++8U87ZQ3BK65a1a9fqVoPaJZAWb7/9dlps27ZtDB/M7kKvLoFbbrmleXPZsmXpkwPGzy6lPAFlBiS2bNmSD5177rn5hKeeeiofaj5neSe9kA0N/xvKN9//uDtyzTXXnHHGGc2du+++Oz3ZeX3qqacOuf9uWkzz2ps5c2busFSRWkrrI488cuiDtqdMmVKOzp49O28uWbKknFCOCnY/77zxxhv5tS9x4okn/uEPf2iesydQmhn1JZDYvn17ufn+x90R/xLIpCeidvcW418CzZsf/vCH+8b6EiifEKT16aefzvjZpZSnpCzK/6wo/Xn99dfL0ZNPPrkczTvpCTv88MObO5Z0KH1Yu+nc3EMo195DDz103HHH5c3+/v5XX311aMdnZ/PmzWnx7rvv5s0bb7yx/H8O007ab55vsYdGsrMn0Cw5L7q9BMraeffTt7NLoG+YxsE9hRFeAmla58/DxvAS+NKXvvT5z3++HGL87GrKU/LMM8/kCyA98XmzHFq9evWcOXPkLvnkQvN8y8DAwCOPPCKbcr5z9xbT/NTvkksuyWWmup599tmh4csjH2qWU9pucsEFF8hpgj3U3Pn2t79tT9hDKH/x+CXgjJ+RXAIdd1qPfwmUQprN5HVpPjOKS+Css85avHjxbbfdlm8yfnY15Skpi61bt+Z12dmyZUu6IPO67Nvn0u4UXnzxxXL0sssuS3+ecsop8+fPb57j3L3FlGvvU5/6VPlfs0+aNClfe+lqzDvNckr/+QtEieYXEMppQvPlr3ylKN9Mn1Q6d2w9Y3IJ5O+ZOePHuQQOOOCActoe+ET4l0ApZO+995Z/55FLIL3TGmp8nCbXXXdd8167DMbP+4u8Ljt5/cYbbzSPpmtm3rx5efGFL3whH3rrrbfKXYSpU6f+7//+79DwlVw+SBO7sydQrr2TTjrp5JNPTos1a9akKvIvUy/j55hjjjnvvPPSYuHChbmoV155pTRmFx1JR++5556hD77CnodQ+tTPv1frse2Vf5/NZvrql0D+xufQ8IvgKC6Bgw466Nxzz02L3/zmN83EPQT/EiiFlM8JxuQSyP/s8xumAu9+djXl2Xr44YenTJly2mmnlU15Ik844YSPfOQjzf30FKZPScqP8KaXs3SBPfnkk+Uuwvr169O/s/333/++++5bsGCBfHz/301baX7l4cILL5w8efI111zz2GOP5depMn4SN9xwQ3qC0ufXpaj0Gfehhx6a7p4HSeKmm26S72YLS5cu3XfffS+//PL0Qpk+zuuvv963I4ODg3qftlP6HPUlkL9/kBndJZCeuPT5/lVXXbXj6XsE/iXQfArSP9exugTmzJmT39FOmzatHGL8TFDK833llVd+85vf3PEgjDvNi1BeE2HXwCXQW9Kbnttvvz2vW3kJMH6qnHrqqfmz4/SJgx7bkXyaoCdBlzT/czk9ZmgU/z7Nd1EwOrgEes7Iy2wU/z4T/xJg/AAAQA9g/AAAQA9g/AAAQA9g/AAAQA9g/ACMnkuG0d2d0TeC7yQDtB7GD8DoGd34AYAhxg9AhDx+fvCDH+y999733ntv2V+yZMnkyZPz7yTOpHc8CxYsaP6mWvkx2fLba0466aQTTjiheceh4d8BccQRR5RNgBbA+AEYPZcM/77IFStWDA3PiQceeCAvzj///LSYP39+Hh5588Ybb7zrrrvy+r8f4oOjZbFmzZqXXnqpuZPX+X+OUO4CsLvD+AEYPWn8HHTQQXn94x//+Kijjtq+fXtzSKT1pk2b8qK5Wdb55rZt24aGf6NaeiOVN2+99db8m1ScOwLs1jB+AEZP83s/g4ODs2bNSu9v9tlnn3LCzJkzb7755qH6FEnr/JZI9svN2h0BdncYPwCjx46f/H/QKSek9cqVK/OiuZkX6e1O/oWeZX/16tV5/fzzzzN+oN0wfgBGjx0/Q8NDovw24jIw7BRZvHixjJNHH320eX7+Pfn2jgDtgPEDMHo6jp933nmn7wPK/wvHTpFyTiZ/nCuvvDLfXLhwYe2OAO3g/wGABRgIxBv+8QAAAABJRU5ErkJggg==>