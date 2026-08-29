# Titanic Survival Prediction

## Project Overview

This project predicts passenger survival on the Titanic through preprocessing and modeling of the Kaggle dataset. A Random Forest model is built for classification, and the core factors influencing survival are identified.

## Dataset

The training set contains 891 passengers with their relevant information, including gender, age, cabin class, fare, and port of embarkation.

Three columns were found to have missing values:
- **Embarked**: only 2 missing values, removed by dropping the corresponding rows.
- **Age**: a small number of missing values, filled with the median.
- **Cabin**: more than 70% missing, labeled as `Unknown` to preserve the signal of whether a cabin record exists.

## Feature Engineering

Non-numeric values were converted to numeric form for modeling:
- Gender mapped as `male=0, female=1`.
- Title (e.g., Mr, Mrs, Miss) extracted from passenger names to reflect social status and marital condition.
- Family size (`SibSp + Parch + 1`) and whether the passenger was alone (`IsAlone`) were constructed.
- Age was binned into four groups: `0-12 (Child), 12-30 (Young), 30-50 (Middle), 50-80 (Senior)` to capture non-linear relationships with survival.

## Exploratory Data Analysis (EDA)

Based on the correlation heatmap and preliminary checks, three features showed stronger correlation with survival: `Sex`, `Pclass`, and `AgeGroup`. Therefore, EDA focused on these three variables.

Key observations from the bar charts:
- **Gender**: female survival rate is approximately 74%, male only about 18%.
- **Cabin Class**: first-class passengers had the highest survival rate.
- **Age Group**: children (<12) had significantly higher survival, consistent with the "women and children first" policy and the closer proximity of first-class cabins to lifeboats.

![Survival by Sex](images/eda_sex.png)

![Survival by Pclass](images/eda_pclass.png)

![Survival by Age Group](images/eda_agegroup.png)

## Encoding Categorical Variables

All categorical variables, including `Sex`, `Embarked`, `Title`, and `AgeGroup`, were encoded into numeric form, as scikit-learn models require numerical input.

## Modeling and Evaluation

1. Feature matrix `X` and target variable `y` (Survived) were defined.
2. The data was split into 80% training and 20% validation sets. The validation set was not used during training to ensure an unbiased evaluation.
3. A Random Forest model (100 trees) was trained on the training set. Random Forest was chosen for its stability on small datasets like Titanic and its ability to output feature importance.
4. The model was evaluated on the validation set:
   - **Confusion Matrix**: The model identifies non-survivors with approximately 82% accuracy and survivors with approximately 69% accuracy. It tends to be conservative, more likely to predict non-survival.
   - **Feature Importance**: The top three features are `Sex`, `Title`, and `Pclass`. The high ranking of `Title` confirms that this feature engineering step was effective.

Validation accuracy: **0.7697**

![Confusion Matrix](images/confusion_matrix.png)

![Feature Importance](images/feature_importance.png)

## Conclusion

The model achieves an accuracy of 76.97% on the validation set, confirming that gender, cabin class, and social status are key factors influencing survival on the Titanic. Further improvements could be made through cross-validation, hyperparameter tuning, or constructing interaction features such as `Pclass × Sex`.