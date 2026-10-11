# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:19:50 2026

@author: Malik Syed
"""
============================================================
CALORIES CONSUMED vs WEIGHT GAINED
SIMPLE LINEAR REGRESSION WITH TRANSFORMATIONS
============================================================
------------------------------------------------------------
BUSINESS UNDERSTANDING
------------------------------------------------------------
1️ Business Problem Statement:
Nutritionists and healthcare professionals want to understand
how daily calorie consumption affects weight gain.
2️ Business Objective:
- Identify the strength of relationship between calorie intake
and weight gain
- Build a predictive model for weight gain
- Improve diet planning recommendations
3️ Motivation:
Understanding calorie–weight relationship helps:
- Prevent obesity
- Design controlled diet plans
- Estimate risk of excessive weight gain
4️ Constraints:
- Human metabolism varies
- Data size may be small
- Linear assumptions may not always hold
- Outliers may affect regression
5️ Success Criteria:
Business Success:
- Accurate prediction of weight gain
- Actionable dietary insights
ML Success:
- Lower RMSE
- Strong correlation
- Stable performance on test data
------------------------------------------------------------
DATA UNDERSTANDING
------------------------------------------------------------

'''

Feature Name	Description	Type	Business Relevance
wt_gained	Weight gained (grams)	Numeric	Target variable
cal_consumed	Calories consumed per day	Numeric	Key predictor
'''			
# ============================================================
# CALORIES CONSUMED vs WEIGHT GAINED
# EXPLORATORY DATA ANALYSIS WITH INFERENCE
# ============================================================

# ------------------------------------------------------------
# STEP 1: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# STEP 2: LOAD THE DATASET
# ------------------------------------------------------------
cal = pd.read_csv("c:/360DG/Datasets/calories_consumed.csv")
cal.columns = ["wt_gained", "cal_consumed"]

print("First 5 Rows:\n", cal.head())

# ------------------------------------------------------------
# STEP 3: BASIC EDA
# ------------------------------------------------------------
print("\nData Types:\n", cal.dtypes)
print("\nSummary Statistics:\n", cal.describe())


# ------------------------------------------------------------
# BUSINESS MOMENT DECISIONS
# ------------------------------------------------------------

# Mean
print("\nMean:\n", cal.mean())
'''
Inference:
• Average calorie intake represents typical daily consumption.
• Average weight gain shows normal gain pattern.
• If both means are high → population likely in calorie surplus.
• Indicates calories may be contributing to weight gain.
'''

# Variance
print("\nVariance:\n", cal.var())
'''
Inference:
 High variance in calories → different eating habits.
 High variance in weight gain → metabolic differences.
 Larger spread means predictions may vary across individuals.
'''

# Standard Deviation
print("\nStandard Deviation:\n", cal.std())
'''
Inference:
 Shows average deviation from mean.
 Lower value → stable population behavior.
 Higher value → more fluctuation in diet and weight gain.
'''

# Skewness
print("\nSkewness:\n", cal.skew())
'''
Inference:
 Skew ≈ 0 → Symmetric distribution.
 Positive skew → few individuals consume very high calories.
 Positive skew in weight  cal consumtion too → few individuals gain extreme weight.
'''

# Kurtosis
print("\nKurtosis:\n", cal.kurtosis())
'''
Inference:
 High kurtosis → presence of extreme values.
 Low kurtosis → uniform spread.
 Extreme calorie or weight values may influence regression.
'''

# Correlation
print("\nCorrelation Matrix:\n",
      np.corrcoef(cal.cal_consumed, cal.wt_gained))
'''
Inference:
 Correlation close to +1 → Strong positive relationship.
 Confirms higher calories → higher weight gain.
 Suitable for linear regression modeling.
'''


# ------------------------------------------------------------
# STEP 4: UNIVARIATE ANALYSIS
# ------------------------------------------------------------

# Histogram - Weight Gained
plt.figure(figsize=(6,4))
plt.hist(cal.wt_gained)
plt.title("Weight Gained Distribution")
plt.xlabel("Weight Gained")
plt.ylabel("Frequency")
plt.show()

'''
Inference:
Most individuals gained lower to moderate weight (clustered in the lower range).
The distribution appears positively skewed (right-skewed).
A few individuals show very high weight gain, creating a long right tail.
These extreme values may act as outliers and can influence regression results.
'''

# Histogram - Calories Consumed
plt.figure(figsize=(6,4))
plt.hist(cal.cal_consumed)
plt.title("Calories Consumed Distribution")
plt.xlabel("Calories Consumed")
plt.ylabel("Frequency")
plt.show()

'''
Inference:
Calorie intake is spread across a moderate to high range.
Most individuals consume calories within a normal daily intake band.
The distribution shows a slight right skew, indicating a few individuals consume very high calories.
Presence of high-calorie values suggests potential overconsumption cases.'''

# ------------------------------------------------------------
# BOX PLOT (OUTLIER DETECTION)
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=cal[['cal_consumed','wt_gained']], orient='h')
plt.title("Boxplot of Numerical Features")
plt.show()

'''
Inference:
Both Calories Consumed and Weight Gained show a reasonable spread.
The data points lie within the whiskers — no extreme outliers detected.
Calories consumed has a wider range compared to weight gained.
Median values appear centrally positioned, indicating stable distributions.
'''


# ------------------------------------------------------------
# BIVARIATE ANALYSIS (SCATTER PLOT)
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x='cal_consumed', y='wt_gained', data=cal)
plt.title("Calories Consumed vs Weight Gained")
plt.xlabel("Calories Consumed")
plt.ylabel("Weight Gained")
plt.show()

'''
Inference:
Clear positive linear relationship observed.
As calories consumed increase, weight gained also increases.
Data points follow an upward trend, indicating a strong correlation.
No major irregular patterns or clustering observed.
'''


# ------------------------------------------------------------
# CORRELATION HEATMAP
# ------------------------------------------------------------
plt.figure(figsize=(5,4))
sns.heatmap(cal.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

'''
Inference:
The correlation between calories consumed and weight gained is 0.95, which indicates a very strong positive relationship.
This means as calorie intake increases, weight gain increases significantly.
The value is close to +1, showing a strong linear association.
No negative relationship observed.'''


# ------------------------------------------------------------
# PDF & CDF ANALYSIS
# ------------------------------------------------------------
for col in ['cal_consumed','wt_gained']:
    plt.figure(figsize=(12,4))

    # PDF
    plt.subplot(1,2,1)
    sns.kdeplot(cal[col], fill=True)
    plt.title(f'PDF of {col}')

    # CDF
    plt.subplot(1,2,2)
    sorted_vals = np.sort(cal[col])
    y = np.arange(len(sorted_vals))/len(sorted_vals)
    plt.plot(sorted_vals, y)
    plt.title(f'CDF of {col}')

    plt.show()

'''
Inference:
PDF (Probability Density Function) fig-1
The distribution appears slightly right-skewed.
Most individuals consume calories in the mid-range (around 1800–2800).
A few individuals consume very high calories, forming the right tail.
Indicates moderate variability in daily calorie intake.
CDF (Cumulative Distribution Function)
The curve increases steadily, showing gradual accumulation.
Around 80% of individuals consume below approximately 3000 calories.
Only a small percentage consume extremely high calories (above 3500–4000).

PDF (Probability Density Function) fig-2

The distribution is positively skewed (right-skewed).
Most individuals gained weight in the lower to moderate range.
A small number show high weight gain, forming the right tail.
Indicates that extreme weight gain cases are limited but present.
CDF (Cumulative Distribution Function)
The curve rises quickly in the lower range, meaning many individuals gained smaller amounts of weight.
Around 70–80% of individuals gained below approximately 600–700 grams.
Only a small percentage experienced very high weight gain (above 900–1000 grams).
'''
# ------------------------------------------------------------
# FINAL EDA SUMMARY
# ------------------------------------------------------------
"""
FINAL SUMMARY:

 Strong positive relationship observed between calorie intake and weight gain.
 Distribution mostly stable with few extreme cases.
 Correlation confirms calories significantly impact weight.
 Linear regression is appropriate for modeling.
 Useful for diet planning and obesity prevention strategies.
"""
#CALORIES CONSUMED VS WEIGHT GAINED
#DATA PREPROCESSING FOR REGRESSION


#STEP 1: IMPORT REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy.stats import skew
from feature_engine.outliers import Winsorizer


#STEP 2: LOAD DATASET

cal = pd.read_csv("C:/13- Linear Regression/calories_consumed.csv")
cal.columns = ["wt_gained", "cal_consumed"]

print("Initial shape:", cal.shape)
print(cal.head())

#STEP 3: BASIC CLEANING

print("\nMissing Values Before Treatment:\n", cal.isnull().sum())

'''
Inference:
Dataset contains only numerical variables.
No identifier column present
If missing values exist -> must be handled.
If zero -> dataset is clean.
'''

#STEP 4: MISSING VALUE TREATMENT if applicable
# Using Median Imputation (robust to skewness)

for col in cal.columns:
    cal[col].fillna(cal[col].median(), inplace=True)
    
print("\nMissing Values After Treatment:\n", cal.isnull().sum())
'''
Inference:
Median used because it is robust to extreme values.
Prevents distortion in regression coefficients.
Suitable for numerical regression datasets.
'''

#STEP 5: DUPLICATE REMOVAL IF APPLICABLE

cal.drop_duplicates(inplace=True)
print("\nShape After Removing Duplicates:",cal.shape)
'''
Inference:
Removes repeated observations.
Prevents model from learning duplicated patterns.
Improves generalization capability.
'''

#STEP 6: OUTLIER DETECTION

plt.figure(figsize=(8,5))
sns.boxplot(data=cal, orient='h')
plt.title("Boxplot Before Treatment")
plt.show()
'''
Inference:
Check for extreme calorie consumption.
Check for extreme weight gain.
outliers can heavily influence regression slope.
'''

#STEP 7: OUTLIER TREATMENT (WINSORIZATION)

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['cal_consumed', 'wt_gained']
)

cal[['cal_consumed','wt_gained']] = winsor.fit_transform(
    cal[['cal_consumed', 'wt_gained']]
)

plt.figure(figsize=(8,5))
sns.boxplot(data=cal, orient='h')
plt.title("Boxplot After Winsorization")
plt.show()
'''
Inference:
Extreme values capped using IQR Method.
Reduces impact of abnormal observations.
Improves regression stability.
Recommended for linear regression.
'''

#STEP 8 : SKEWNESS CHECK

print("\nSkewness:\n", cal.skew())
'''
Inference:
Skewness > 1 -> Strong skew (log transformation may help)
Mild skew -> transformation optional
helps decide model transformation stage
'''

#FINAL PREPROCESSING SUMMARY

'''
FINAL DATA PREPROCESSING SUMMARY:

Dataset validated and cleaned.
Missing values handled using median.
Duplicates removed.
Outliers treated using IQR-based winsorization.
Skewness evaluated.

Dataset is now ready for regression modeling.
'''

#MODEL DEVELOPMENT - SIMPLE LINEAR REGRESSION

#LET US APPLY TO VARIOUS MODELS AND CHECK FEASIBILITY

import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# 1. SIMPLE LINEAR REGRESSION

model1 = smf.ols('wt_gained ~ cal_consumed', data=cal).fit()
pred1 = model1.predict(cal)

rmse1 = np.sqrt(np.mean(cal.wt_gained - pred1)**2)
print("SLR RMSE:", rmse1)
#103.30
model1.summary()

'''

#R-squared = 0.897 > 0.80, Model is very strong
#p = 0 < 0.05 hence acceptable
#beta-0 = -625.75
#beta-1 = 0.4202

Goal of the Model
We are trying to predict wt_gained (Dependent Variable)
using cal_consumed (Independent Variable).
1. Model Fit (Goodness of Fit)
R-squared = 0.897(applicable for MLR)
“Adjusted R² tells us whether adding more variables truly improves the model or just artificially increases R².”
→ About 89.7% of the variation in wt_gained is explained by cal_consumed.
This means the model fits the data very well.
Adjusted R-squared = 0.888
→ Adjusted for number of predictors; still very high.
Since there is only one predictor, overfitting is not a concern.

purpose of F-statistic
Does the independent variable(s) collectively explain the dependent variable?
Or is the relationship happening just by chance?
F-statistic = 104.3
An F-value of 104.3 is very large.
This means the model explains much more variation in weight gain than random chance.
It indicates that calories consumed significantly improves prediction of weight gained.


 Model is statistically significant.
 It performs much better than a model with no predictor.
 Calories consumed significantly explains weight gain.

2. Coefficient Interpretation
Variable	Coefficient	Interpretation
Intercept	-625.75	When calories = 0, predicted weight gain is -625g (not practical, just mathematical baseline).
cal_consumed	+0.4202	For every 1 unit increase in calories, weight increases by 0.42 units.  Strong positive relationship.

P-value for cal_consumed = 0.000
 Highly statistically significant.

This confirms:
Higher calorie intake leads to higher weight gain.

3. Residual Analysis

The Durbin-Watson statistic checks whether residuals are independent.

DW Value	Meaning
≈ 2	 No autocorrelation (Ideal)
< 1.5	 Positive autocorrelation
> 2.5	 Negative autocorrelation
1.5–2.5	 Acceptable

Durbin-Watson = 2.537

 Slightly above 2 but still acceptable.
 No serious autocorrelation problem.

Normality Check

Jarque-Bera p-value = 0.541 (> 0.05)
 Residuals are approximately normally distributed.
 Regression assumption satisfied.

In One Sentence :
“Calories consumed has a strong and statistically significant impact on weight gained, and this model explains nearly 90% of the variation in weight. The regression assumptions are largely satisfied, so the model is reliable for prediction.”
'''

# 2 LOG MODEL (log X)

model2 = smf.ols('wt_gained ~ np.log(cal_consumed)', data=cal).fit()
pred2 = model2.predict(cal)

rmse2 = np.sqrt(np.mean((cal.wt_gained - pred2)**2))
print("Log-X Model RMSE:", rmse2)

model2.summary()

'''
R-squared = 0.808 = 0.8, there is scope for improvement
p = 0.000 < 0.05 hence acceptable
beta-0 = -6955.65
beta-1 = 948.37
'''

# 3 . EXPONENTIAL MODEL (LOG Y)
model3 = smf.ols('np.log(wt_gained) ~ cal_consumed', data=cal).fit()
pred3 = np.exp(model3.predict(cal))

rmse3 = np.sqrt(np.mean((cal.wt_gained - pred3)**2))
print("Exponential Model RMSE:", rmse3)

model3.summary()

# 4. POLYNOMIAL MODEL

model4 = smf.ols(
    'np.log(wt_gained) ~ cal_consumed + I(cal_consumed**2)',
    data=cal).fit()

pred4 = np.exp(model4.predict(cal))
rmse4 = np.sqrt(np.mean((cal.wt_gained - pred4)**2))
print("polynomial model rmse:", rmse4)

model4.summary()

#MODEL COMPARISON

results = pd.DataFrame({
    "Model" : ["SLR", "LOG-X", "Exponential", "Polynomial"],
    "RMSE" : [rmse1, rmse2, rmse3, rmse4],
    "R_squared": [
        model1.rsquared,
        model2.rsquared,
        model3.rsquared,
        model4.rsquared
    ]
    
})

print("\nModel Comparison:\n",results)

# SELECT BEST MODEL

best_model_name = results.sort_values("RMSE").iloc[0]["Model"]
print("\nBest Model based on RMSE:", best_model_name)

# TRAIN TEST VALIDATION USING BEST MODEL SLR

train, test = train_test_split(cal, test_size=0.3, random_state=42)

final_model = smf.ols('wt_gained ~ cal_consumed', data=train).fit()

train_pred = final_model.predict(train)
test_pred = final_model.predict(test)

train_rmse = np.sqrt(np.mean((train.wt_gained - train_pred)**2))
test_rmse = np.sqrt(np.mean((test.wt_gained - test_pred)**2))

print("\nTrain RMSE:", train_rmse)
print("\nTest RMSE:", test_rmse)
