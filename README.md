Landmine Detection Using Machine learning

A CatBoost-Based Framework for Detecting Mines, and IEDs

1. Introduction

This project implements a complete machine-vision workflow for detecting landmines, improvised explosive devices (IEDs),using features derived from voltage, height, and soil-type measurements.
The system provides:

Data loading, cleaning, and preprocessing

Feature engineering for voltage, height, and soil-type

Model training using CatBoost with K-Fold cross-validation

Command-line and API-based inference

Dockerised deployment for reproducibility

The project is intended for research, prototyping, and applied machine learning in defence and humanitarian contexts.
## Project Structure

<details>
<summary>Click to expand</summary>

.
├── app/ # Flask application for serving predictions
├── catboost_info/ # CatBoost training logs and metadata
├── Dockerfile # Container specification
├── Mine_Dataset.xls # Dataset with normalized features
├── model_depth_10_lr_0.02_iter_200.bin # Trained CatBoost model
├── notebooks/
│ └── train_notebook.ipynb # Exploratory model training notebook
├── Pipfile / Pipfile.lock # Python environment definitions
├── src/
│ ├── train.py # Training pipeline with K-Fold CV
│ ├── predict.py # CLI prediction script
│ ├── predict_flask.py # Flask API wrapper
│ ├── predict_test.py # API test harness
│ └── preprocessing/ # Feature extraction and preprocessing utilities
└── README.md # Project documentation
</details>

4. Installation
3.1 Requirements

Python 3.10+

Pipenv (recommended) or pip

macOS / Linux / Windows

CPU-only operation supported
3.2 Using Pipenv (Recommended)
pip install pipenv
pipenv install
pipenv shell

. Dataset & Preprocessing
4.1 Dataset

The dataset is located at:

Mine_Dataset.xls


It contains:

Voltage, height, and categorical soil type
Dataset

The dataset used for training and evaluation is contained in:

Mine_Dataset.xls (from:KAHRAMAN, H. (2018). Land Mines [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54C8Z.)

4.1 Dataset Contents
| Variable Name | Role    | Type       | Description                                                                                                                                        | Units | Missing Values |
| ------------- | ------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | -------------- |
| **V**         | Feature | Continuous | Output voltage value of the FLC sensor resulting from magnetic distortion.                                                                         | V     | No             |
| **H**         | Feature | Continuous | Height of the sensor above the ground.                                                                                                             | cm    | No             |
| **S**         | Feature | Continuous | Soil type category. Six soil types based on moisture conditions: dry & sandy, dry & humus, dry & limy, humid & sandy, humid & humus, humid & limy. | —     | No             |
| **M**         | Target  | Integer    | Mine type: five different landmine classes commonly encountered in field conditions.                                                               | —     | No             |



The preprocessing pipeline is designed to prepare the landmine dataset for machine learning models and incorporates both numerical and categorical features informed by exploratory data analysis (EDA). Key steps include:

Normalization and Type Conversion

String representations of numerical measurements (e.g., voltage and height) are converted to float.

Values are normalised to a consistent range (0–1) to ensure comparability across features.

Soil-Type Categorical Encoding

The Soil_type variable, originally in normalized increments (0, 0.2, …, 1.0), is transformed into a categorical feature soil_type_cat with 6 categories.

This captures discrete soil characteristics relevant to mine detection and allows models to learn soil-dependent patterns.

Feature Engineering & IR–Visible Fusion

Combines Voltage, Height, and soil_type_cat to capture interactions between sensor readings and terrain.

Enables models to differentiate mine types (Anti-Tank, Anti-Personnel, Booby-Trapped, M14) and background/no-mine cases.

Insights from global landmine contamination patterns (Ukraine, DR Congo, Sudan, Myanmar, Yemen, Angola) guided the choice of features that reflect environmental risk factors and detectability.

Noise Reduction & Outlier Handling

Outliers in sensor readings are identified and removed to improve model stability.

Missing values (if any) are imputed using median values to maintain consistent feature dimensions.

Boxplots, violin plots, and PCA visualisations from EDA are used to confirm feature distributions and separability between classes.

Target Feature Analysis

Mine_type distribution was analysed to identify class imbalances and inform evaluation metrics.

Violin plots and PCA projections revealed meaningful separation between classes, validating the selected features.

Validation and Data Splits

Dataset is split into training, validation, and test sets with reproducible random shuffling.

K-Fold cross-validation ensures robust model evaluation and reduces overfitting risk.

Outcome: The resulting processed dataset captures all critical patterns needed for CatBoost and other ML models, balancing numerical precision with categorical interpretability, and is informed by both feature-level analysis and global landmine contamination context.




5. Training the Model
5.1 Script-Based Training

Run:

python src/train.py


Pipeline:

Loads Mine_Dataset.xls and normalises data

Performs feature engineering

Splits dataset into training, validation, and test sets

Trains CatBoostClassifier with K-Fold cross-validation (5 splits)

Evaluates final AUC on the test set

Saves trained model and vectoriser:

model_depth_10_lr_0.02_iter_200.bin

5.2 Training Parameters

Depth: 10

Learning rate: 0.02

Iterations: 200

Loss function: Multi-class cross entropy

Features: Voltage, Height, soil_type_cat

5.3 Notebook-Based Training

Exploratory training can be performed in:

notebooks/train_notebook.ipynb

6. Model Inference
6.1 Command-Line Prediction
python src/predict.py --input path/to/sample.json

6.2 API Test Harness
python src/predict_test.py


Sends a JSON POST request to the Flask API at http://localhost:9898/predict_flask

Maps predicted class numbers to human-readable labels

Prints class probabilities and confidence

6.3 Flask API

The Flask API is implemented in src/predict_flask.py:

Endpoint: /predict_flask

Method: POST

Input: JSON dictionary with features (excluding Mine_type)

Output: JSON containing predicted class, class probabilities, and confidence

Example:

curl -X POST http://localhost:9898/predict_flask \
     -F '{"Voltage":0.24,"Height":0.72,"soil_type_cat":1}'

7. Docker Deployment

The project includes a Dockerfile for containerised deployment.

7.1 Build the Docker Image
docker build -t landmine-detector .

7.2 Run the Container
docker run -p 9898:9898 landmine-detector


The Flask API will be accessible at:

http://localhost:9898/predict_flask

7.3 Dockerfile Highlights

Python 3.12 slim image

Pipenv system-wide installation

Dependency caching for faster builds

Copies trained model and API scripts into container

Uses Gunicorn to serve Flask app:

ENTRYPOINT ["gunicorn","--bind=0.0.0.0:9898","predict_flask:app"]

8. Reproducibility

All hyperparameters and random seeds are fixed in train.py

Pipfile.lock ensures identical environments

Model binary and DictVectorizer are committed for reference

Preprocessing logic is version-controlled under src/preprocessing/
9. Future Improvements

Deep-learning based thermal–visible fusion

Soil anomaly segmentation using CNNs

SHAP-based interpretability

Integration with UAVs for aerial survey

Expansion of real-world field data

Embedded deployment on edge devices (Raspberry Pi, Jetson Nano)

10. Author

Mnoneleli Meshark Govuzela
Email: govuzelamm@gmail.com

Port Elizabeth, South Africa

11. Licence

Add your preferred licence (e.g., MIT, Apache-2.0, GPLv3) for public release.

This version fully documents your:

Training pipeline (train.py)

Flask API (predict_flask.py)

API test (predict_test.py)

Docker deployment (Dockerfile)

Dataset and preprocessing logic

I can also create a diagram showing the full workflow from dataset → preprocessing → training → API → Docker if you want a visual for the README.

Do you want me to do that?
