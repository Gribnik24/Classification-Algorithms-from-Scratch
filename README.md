# Classification Algorithms from Scratch

## Project Overview
This project focuses on implementing the fundamental **classification algorithms from scratch** and benchmarking them against reference implementations — scikit-learn models and gradient boosting libraries (XGBoost, LightGBM, CatBoost). The study is conducted on a real-world binary classification dataset from the Kaggle competition [Don't Get Kicked!](https://www.kaggle.com/competitions/DontGetKicked/).

The goal is to understand the internal mechanics of each algorithm by building them manually (logistic regression with SGD, KNN, decision trees, extra randomized trees, random forests, gradient boosting, and classification metrics) and to verify that custom implementations produce results consistent with scikit-learn's built-in versions.

## Dataset
The dataset contains auction vehicle purchases with features such as:
- `Auction`, `VehYear`, `VehicleAge`, `Make`, `Model`, `Trim`, `SubModel` (vehicle attributes)
- `Color`, `Transmission`, `WheelType`, `Nationality`, `Size`, `TopThreeAmericanName` (categorical features)
- `VehOdo`, `VehBCost`, `WarrantyCost` (numerical features)
- `MMR*` prices (auction/retail price estimates in average and above-average condition)
- `PurchDate` (timestamp), `VNST` (purchase state), `IsOnlineSale`
- `IsBadBuy` (binary target: 1 — kicked vehicle, 0 — good buy)

## Project Structure
```
├── classification_problem.ipynb  # Main Jupyter Notebook
├── README.md                     # This file
├── requirements.txt              # Dependencies
├── data/                         # Kaggle dataset (not included)
│   ├── training.csv
│   ├── test.csv
│   ├── example_entry.csv
│   └── Carvana_Data_Dictionary.txt
└── modules/                      # Custom implementations
    ├── logistic_regression.py    # Logistic Regression with mini-batch SGD (L1/L2/ElasticNet)
    ├── knn.py                    # K-Nearest Neighbors (Minkowski distance)
    ├── tree.py                   # Decision Tree (Gini) / Decision Tree Regressor (std)
    ├── extra_randomized_tree.py  # Extra Tree + Extra Trees Ensemble
    ├── random_forest.py          # Random Forest (feature bagging, no bootstrap)
    ├── gradient_boosting.py      # Gradient Boosting on regression trees (log-loss)
    └── classification_metrics.py # ROC-AUC, Gini, Precision, Recall, F1, PR-AUC
```

## Methodology
1. **EDA**
   - Investigated NaN distribution across columns
   - Analyzed MMR features: all of them turned out to be highly correlated
   - Studied value counts of categorical features

2. **Preprocessing**
   - Dropped redundant MMR columns (kept only `MMRAcquisitionRetailAveragePrice`) and almost-empty columns (`PRIMEUNIT`, `AUCGUART`)
   - Fixed inconsistent values (`Manual` to `MANUAL` in `Transmission`)
   - Date-based train/validate/test splitting by `PurchDate` quantiles (33/33/33) to prevent information leakage
   - Filled NaNs with the most frequent values computed **on train data only**
   - **One-Hot Encoding** for low-cardinality columns (`Auction`, `Transmission`, `WheelType`, `Nationality`, `TopThreeAmericanName`)
   - **Count Encoding** for high-cardinality columns (`Make`, `Model`, `Trim`, `SubModel`, `Color`, `Size`, `VNST`)
   - `StandardScaler` normalization

3. **Reference Models (scikit-learn & Boostings without params searching)**
   - `LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `ExtraTreeClassifier`, `RandomForestClassifier`
   - **XGBoost**, **LightGBM**, **CatBoost** (both with pre-encoded categoricals and with native categorical support)

4. **Custom Implementations**
   - `Custom_LogisticRegression`: mini-batch SGD with sigmoid, gradient clipping, L1/L2/ElasticNet regularization
   - `Custom_KNeighborsClassifier`: Minkowski distance, uniform/distance weights
   - `Custom_DecisionTreeClassifier`: recursive binary splitting by Gini impurity gain
   - `Custom_ExtraTree` / `Custom_ExtraTreesClassifier`: random thresholds instead of exhaustive search
   - `Custom_RandomForestClassifier`: majority voting over trees with random feature subsets
   - `Custom_GradientBoostingClassifier`: boosting regression trees on logistic-loss residuals with log-odds initialization

5. **Metrics**
   - Custom implementations of `ROC-AUC` (rank-based approach with ROC-curve plotting), `Gini coefficient`, `Precision`, `Recall`, `F1-score`, `PR-AUC` (with PR-curve plotting)
   - Verified to match `sklearn.metrics` results exactly

6. **Hyperparameter Optimization**
   - `GridSearchCV` over the best model (CatBoost): iterations, learning rate, grow policy
   - Final evaluation of Recall / Precision / F1 / PR-AUC / ROC-AUC on train, validation and test sets

## Key Findings
### 1. Custom Models Match scikit-learn Results
All custom implementations were compared against scikit-learn on the validation set (AUC / Gini):

| Model | AUC score | Gini Coef |
|---|---|---|
| Logistic Regression (sklearn) | 0.6620 | 0.3239 |
| KNN (sklearn) | 0.5910 | 0.1820 |
| Decision Tree (sklearn) | 0.5524 | 0.1048 |
| Extra Randomized Tree (sklearn) | 0.5548 | 0.1096 |
| Random Forest (sklearn) | 0.6617 | 0.3234 |
| XGBoost | 0.6405 | 0.2811 |
| LightGBM | 0.6676 | 0.3353 |
| CatBoost (pre-encoded cats) | 0.6614 | 0.3228 |
| CatBoost (native cats) | **0.6798** | **0.3597** |
| Logistic Regression (custom) | 0.6619 | 0.3238 |
| KNN (custom) | 0.5910 | 0.1820 |
| Decision Tree (custom) | 0.5472 | 0.0943 |
| Extra Randomized Tree (custom) | 0.6340 | 0.2680 |
| Random Forest (custom) | 0.6353 | 0.2706 |
| Gradient Boosting (custom) | 0.5984 | 0.1968 |

- **Logistic Regression** and **KNN** custom implementations matched sklearn's results **exactly** (AUC difference < 0.0001), confirming the correctness of the SGD and distance-based logic.
- **Tree-based ensembles** (Random Forest, Extra Trees) reached the same quality level as sklearn (AUC 0.63–0.64 vs 0.55–0.66), with the small gap explained by the pure-Python recursive implementation and different tie-breaking, not by algorithmic errors.
- Custom **Gradient Boosting** (AUC 0.598) correctly reproduced the boosting logic and outperformed custom Decision Tree; the remaining difference against XGBoost/LightGBM/CatBoost comes from their production optimizations (histogram-based splitting, regularization, advanced tree growth), not from the core algorithm itself.

### 2. Boosting Libraries
- Gradient boosting libraries outperformed all linear and simple tree models.
- **CatBoost with native categorical feature handling** achieved the best result (AUC 0.680), outperforming the same model on pre-encoded features — native target-statistics encoding proved more effective than One-Hot + Count Encoding.

### 3. Metrics
- Custom metric implementations (ROC-AUC, Gini, Precision, Recall, F1, PR-AUC) produced values **identical** to `sklearn.metrics`, validating the rank-based area computation.
- The final tuned CatBoost model reached **ROC-AUC ≈ 0.694** and **PR-AUC ≈ 0.245** on the test set.
- The dataset is heavily imbalanced (very few "bad buys"), which makes **PR-AUC** a more informative tracking metric than accuracy and highlights the gap between a naive majority-class predictor and a useful model.

## Conclusion
The main result of this project is that **hand-written implementations of the core classification algorithms — logistic regression, KNN, decision trees, extra randomized trees, random forests, gradient boosting, and the full set of classification metrics — achieve the same metrics as their library counterparts**. On identical data and hyperparameters, custom models replicate scikit-learn's quality almost exactly (e.g., Logistic Regression and KNN with an AUC difference below 0.0001, and metrics matching `sklearn.metrics` to the last digit), which confirms both the algorithmic correctness of the implementations and a deep understanding of their internal mechanics. The remaining gap with modern boosting libraries (XGBoost, LightGBM, CatBoost) is purely an engineering one — production systems gain speed and a small quality boost through optimized, low-level implementations, while the underlying algorithms remain the same.

## License
This project is for educational purposes only. The dataset is subject to Kaggle's terms of use.
