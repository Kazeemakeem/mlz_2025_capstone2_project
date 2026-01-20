## 1.0 PROBLEM STATEMENT

1.1 Objective: To develop and deploy a machine learning model that accurately classifies breast mass tumors as either Malignant (cancerous) or Benign (non-cancerous) based on digitized measurements from fine needle aspirate (FNA) images.

1.2 Context: Breast cancer is the most common cancer among women worldwide, accounting for roughly 25% of all cases. Early and accurate diagnosis is critical for effective treatment and improved patient survival rates. Traditionally, classifying these tumors requires expert clinical analysis, which can be time-consuming. Leveraging predictive analytics allows for faster, data-driven screening that supports medical professionals in identifying high-risk cases.

1.3 The Problem: The challenge lies in distinguishing between benign and malignant tumors based on complex cellular characteristics. Manual interpretation of these nuclear features can lead to diagnostic delays or errors. There is a need for a robust, automated classification model that can analyze quantitative features—such as cell nuclei radius, texture, and perimeter—to provide a reliable diagnostic prediction.

## 2.0 DATA DESCRIPTION

This Breast Cancer Dataset(source - "https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset" ) consists of 569 instances with 32 columns. It includes a unique ID, a diagnosis label (M = Malignant, B = Benign), and 30 numerical features derived from digitized images of cell nuclei. These features are categorized into three groups: Mean, Standard Error, and Worst (largest) values.

- Target Variable: diagnosis (Malignant or Benign).
- Numerical Features:
  Radius: Mean of distances from center to points on the perimeter.
  Texture: Standard deviation of gray-scale values.
  Perimeter & Area: Measurements of the cell nuclei size.
  Smoothness: Local variation in radius lengths.
  Compactness, Concavity, & Concave Points: Indicators of the nuclei's boundary complexity.
  Symmetry & Fractal Dimension: Additional morphological characteristics.

## 3.0 KEY EDA INSIGHTS

- Class Distribution: The dataset contains 357 Benign (62.7%) and 212 Malignant (37.3%) cases. While relatively balanced, specialized metrics like Recall and F1-Score are vital to ensure malignant cases (False Negatives) are not missed.

- Strong Feature Correlation: Size-related measurements—specifically radius, perimeter, and area—show extremely high positive correlation with one another. Feature selection or dimensionality reduction may be necessary to address multicollinearity.

- Distinct Feature Distributions: Malignant tumors generally exhibit higher mean values for radius, perimeter, and area compared to benign tumors, providing a clear signal for classification.

- Concavity and Compactness: These features often show a wider distribution and more outliers in malignant cases, suggesting that irregular cell shapes are a strong predictor of cancer.

- Data Integrity: The dataset is well-structured with no missing values, allowing for immediate analysis and preprocessing (e.g., standard scaling) to improve model performance.

## 4.0 MODELING APPROACH

    The modeling approach for the income prediction task employed focused on robust evaluation due to the imbalanced nature of the dataset. I established a baseline using Logistic Regression and evaluated subsequent models using Area Under the Precision-Recall Curve (AUC-PR) as the primary metric. We employed cross-validation throughout the process to ensure model stability and then progressed to more complex algorithms, including Decision Tree, Random Forest and XGBoost. This systematic progression allowed us to compare model performance effectively and select the most suitable algorithm for predicting high-income earners.

## 5.0 RUNNING LOCALLY

    i. Clone the repository
    ii. At the root dir, run pip install uv
    iii. RUN "uv sync --locked"
    iv. RUN "uv run train.py"
    v.  RUN the service with "uv run uvicorn predict:app --host 0.0.0.0 --port 9696 --reload"

## 6.0 RUNNING ON DOCKER

    i. Clone the repository
    ii. If you don't docker Installed on your PC, download Docker Desktop and launch
    iii. build the docker image with the command:
        # docker build -t cancer-detection-prediction .
    iv. run the docker image with the command:
        # docker run -it --rm -p 9696:9696 cancer-detection-prediction

## 7.0 API USAGE EXAMPLE

    While the service is running and on a separate terminal, RUN "uv run serve.py"
    vii. You can also visit http://localhost:9696/docs to test the service on the browser

## 8.0 EXAMPLE REQUEST

curl -X 'POST' \
 'http://localhost:9696/predict' \
 -H 'accept: application/json' \
 -H 'Content-Type: application/json' \
 -d '{
"area_mean": 143.5,
"area_se": 6.802,
"area_worst": 185.2,
"compactness_mean": 0.01938,
"compactness_se": 0.002252,
"compactness_worst": 0.02729,
"concave_points_mean": 0.2012,
"concave_points_se": 0.05279,
"concave_points_worst": 0.291,
"concavity_mean": 0.4268,
"concavity_se": 0.396,
"concavity_worst": 1.252,
"fractal_dimension_mean": 0.04996,
"fractal_dimension_se": 0.000895,
"fractal_dimension_worst": 0.05504,
"perimeter_mean": 43.79,
"perimeter_se": 0.757,
"perimeter_worst": 50.41,
"radius_mean": 6.981,
"radius_se": 0.1115,
"radius_worst": 7.93,
"smoothness_mean": 0.05263,
"smoothness_se": 0.001713,
"smoothness_worst": 0.07117,
"symmetry_mean": 0.106,
"symmetry_se": 0.007882,
"symmetry_worst": 0.1565,
"texture_mean": 9.71,
"texture_se": 0.3602,
"texture_worst": 12.02
}'
