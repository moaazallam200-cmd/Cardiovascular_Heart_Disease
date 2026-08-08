# Cardiovascular Disease Prediction

## Project Overview
This project focuses on predicting the presence of cardiovascular disease in patients using medical examination data. It treats the problem as a binary classification task, where the target variable (`cardio`) indicates either the absence (0) or presence (1) of the disease. The primary objective is to construct a robust predictive model that generalizes well to unseen data, prioritizing critical medical evaluation metrics such as Recall and ROC-AUC.

## Repository Contents
This repository contains the following five files:
* **Python Notebook (`.ipynb`)**: The core workflow containing data exploration, feature engineering, and model training using libraries like pandas, scikit-learn, and matplotlib.
* **Project Documentation (`.docx`)**: A text document providing a detailed overview of the project's logic, pipeline, and findings.
* **Dataset (`cardio_train.csv`)**: The raw dataset consisting of 70,000 patient records and 13 columns (including objective features, examination results, and subjective patient information).
* **`gradient_boosting_model.pkl`**: The exported, fully fitted `HistGradientBoostingClassifier` machine learning pipeline.
* **`logistic_model.pkl`**: The exported, fully fitted `LogisticRegression` machine learning pipeline.

## Methodology

### 1. Data Cleaning & Feature Engineering
* Handled data integrity by verifying the absence of missing values and removing duplicate rows.
* Filtered out biologically impossible outliers, such as negative blood pressures, illogical systolic/diastolic ratios, and extreme height/weight values.
* Engineered new, medically relevant features, including converting age from days to `age_years` and calculating Body Mass Index (`bmi`).
* Dropped the non-predictive `id` column to optimize the feature matrix.

### 2. Exploratory Data Analysis (EDA)
* Validated that the dataset is well-balanced between positive and negative cardiovascular disease cases.
* Utilized seaborn boxplots and countplots to visualize feature relationships, revealing that patients with the disease tend to be older and have higher cholesterol levels.
* Generated a correlation heatmap to analyze linear dependencies across numerical variables.

### 3. Model Training & Evaluation
* Split the data into 80% training and 20% testing sets, using stratification to maintain class balance.
* Constructed a `ColumnTransformer` to apply `StandardScaler` to numerical features and `OneHotEncoder` (dropping the first category) to categorical variables.
* Established a baseline accuracy of 0.5053 using a `DummyClassifier`.
* Evaluated three models using a 5-fold Stratified K-Fold cross-validation: Logistic Regression, Random Forest, and HistGradientBoosting.

## Results
The models were evaluated based on their Accuracy, Recall, and ROC-AUC scores. The **HistGradientBoostingClassifier** emerged as the highest-performing algorithm.

| Model | CV Accuracy | CV Recall | CV ROC-AUC |
| :--- | :--- | :--- | :--- |
| **HistGradientBoosting** | 0.7345 | 0.6884 | 0.8003 |
| **Logistic Regression** | 0.7278 | 0.6643 | 0.7913 |
| **Random Forest** | 0.7009 | 0.6908 | 0.7586 |

## How to Use
1. Clone this repository to your local machine.
2. Ensure you have the required dependencies installed (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`).
3. Run the Jupyter Notebook to step through the EDA and training process.
4. To deploy or test the pre-trained models, load the `.pkl` files using `joblib.load()`.
