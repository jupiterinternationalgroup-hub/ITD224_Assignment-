# Condominium Rental Rate Prediction
### Individual Workstream — Singapore Real Estate Intelligence Platform
**Author:** Tan Tien Hock (Alan) | Student ID: 6968437D
**Module:** ITD224 Applied Data Science Project | **Framework:** CRISP-DM

---

## 1. Project Background

The Singapore Real Estate Intelligence Platform is a team project helping a Singapore real estate agency make better pricing, rental, and customer-targeting decisions using ML. This individual workstream focuses on **predicting monthly condominium rental prices** and identifying key pricing factors, enabling landlords, tenants, and agencies to benchmark competitively with model-backed confidence.

**Business Objective:** Predict monthly Executive Condominium rental prices and identify key pricing factors using property attributes (bedrooms, floor size, lease age, district) and location-convenience features engineered from Singapore's OneMap API (distance to nearest MRT, supermarket, school).

**Dataset:** CondoRental.csv — 10,000 URA (Urban Redevelopment Authority) Executive Condominium rental transactions across postal districts 18-28, enriched with GPS coordinates and amenity distances via OneMap API.

**Success Criteria:** Model achieves a materially lower MAE/RMSE than a naive district-average baseline, with an interpretable feature-importance ranking the agency can act on.

---

## 2. Work Accomplished

### Data Preparation
- Cleaned raw fields: stripped comma-formatted currency (Monthly Rent), converted banded floor-area ranges (URA privacy protection) to numeric midpoints, parsed abbreviated lease dates.
- Imputed 3 missing `No of Bedroom` values (0.03%) using median bedroom count within matching floor-area bands.
- Flagged rent outliers via 1.5xIQR rule — retained as genuine premium-market transactions rather than removed.
- Enriched dataset via OneMap API: GPS coordinates, distance to nearest MRT/supermarket/school, and true building age (via OneMap address lookup + web search, correcting an earlier proxy that mistakenly used lease commencement date).
- Feature selection: correlation + VIF screening reduced 9 candidate features to 8 final model features (dropped `FloorAreaSQM`, a near-duplicate of `FloorAreaSQFT`, r=0.978).

### Modelling
- Compared three algorithms: Linear Regression, Random Forest, XGBoost — each wrapped in a scikit-learn pipeline with median imputation and one-hot encoding.
- Benchmarked against a naive district-average baseline (MAE $559, R² 0.066).
- Validated model ranking via 5-fold cross-validation (not just a single train-test split).
- Tuned all three model families via GridSearchCV; verified tuned models on real held-out test data before accepting.

### Final Model: XGBoost

| Metric | Value | Baseline |
|---|---|---|
| MAE | $456 | $559 |
| RMSE | $622 | $746 |
| R² | 0.352 | 0.066 |
| MAE Improvement | 18.5% | — |

---

## 3. Recommendation and Analysis

### Key Finding — What Drives Rent
Unit size (bedroom count) dominates rent prediction, accounting for roughly three-quarters of total feature importance. Location-convenience features (MRT distance, supermarket distance, distance from CBD, specific postal districts) contribute a meaningful but secondary signal.

### Recommendations by Stakeholder

**Landlords — Competitive Pricing Guidance**
- Anchor asking rent to the model's predicted rent ± MAE band ($456), not gut feel.
- Focus on the dominant driver first: bedroom-count upgrades/reconfigurations carry the most pricing leverage.
- Treat location (MRT/supermarket proximity, CBD distance) as the secondary lever.

**Agencies — Listing Benchmarking & Negotiation**
- Screen every listing pre-publication; flag asking rents that fall materially outside the MAE band.
- Use price-per-sqft vs. MRT-distance analysis to back up pricing claims with data instead of subjective "great location" claims.

**Tenants — Fair, Transparent Pricing**
- Benchmark quoted rent against the district-summary reference table; gaps larger than the MAE band are valid negotiation openings.
- Treat thin-sample districts (e.g., District 24, with only 1 transaction) with scepticism — low volume means unreliable averages.

### Model Limitations
At R² = 0.352, roughly 65% of rent variance remains unexplained. This reflects genuine data constraints, not a modelling flaw:
- Weak feature-target correlations (max r = 0.31 for FloorAreaSQFT; all other features below r = 0.16).
- URA privacy-preserving banding of floor area and lease dates discards real within-band variation.
- Missing unit-quality signals (condition, floor level, view, furnishing) not available in the URA extract.

Predictions should be treated as **directional pricing guidance**, not precise valuations.

---

## 4. AI Ethics Considerations

| Dimension | Discussion |
|---|---|
| **Privacy** | Geocoding uses building-level identifiers only (Project Name, Street Name) — never unit numbers or personal tenant/landlord identities. No PII is collected, stored, or exposed. |
| **Fairness** | GPS coordinates fall back to district centroids when live geocoding is unavailable, which could systematically under-value units in under-served districts. Model outputs should be reviewed for disparate treatment across districts before use as pricing floors/ceilings. |
| **Accuracy** | All fallback cases (GPS/amenity data unavailable) are explicitly logged per-category in an audit table, so no stakeholder mistakes a fallback estimate for a live-data guarantee. |
| **Accountability** | The pipeline is fully reproducible and auditable — every transformation (clean, construct, integrate, format) is visible, not hidden behind a black-box prediction. Recommend quarterly re-validation as new transactions accrue. |
| **Transparency** | Original raw columns are retained alongside engineered ones. The feature-importance ranking and MAE confidence band should be surfaced alongside every predicted rent in deployment, so stakeholders see why a number was produced and how much to trust it. |

---

## 5. Repository Structure

```
/notebook
  Tan_Tien_Hock_Condo_Rental_Regression_v3_onemap_jigtech.ipynb
/data
  Condo_Rental_Cleaned_GPS_Enriched.csv       (final dataset used for modelling, 9,997 rows x 40 columns)
  amenity_sources_audit_report.csv            (data provenance / enrichment audit trail, 9,997 rows x 31 columns)
/visualisation
  condo_rental_map.html                       (interactive Folium map of geocoded rental locations)
README.md
```
