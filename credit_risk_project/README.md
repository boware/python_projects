# Credit Risk Prediction Project

This project focuses on predicting credit risk using machine learning. The goal is to predict whether an applicant will default on a loan based on personal and financial information. The dataset used is from the **Home Credit Default Risk** dataset available on Kaggle.

## Project Structure:

- **/data**: Contains raw and processed datasets.
- **/scripts**: Contains Python scripts for data downloading, preprocessing, feature engineering, model training, and evaluation.
- **/notebooks**: Contains Jupyter Notebooks for data exploration, preprocessing, model training, and evaluation.
- **requirements.txt**: List of Python dependencies.
- **credit_risk_model.pkl**: The trained model for predicting credit risk.

## How to Run:

### 1. Install Dependencies
To get started, clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download Data
Use the download_data.py script to download the dataset from Kaggle:
```bash
python scripts/download_data.py
```

### 3. Data Preprocessing
Run the data preprocessing script:
```bash
python scripts/data_preprocessing.py
```

### 4. Feature Engineering
Run the feature engineering script:
```bash
python scripts/feature_engineering.py
```

### 5. Train the Model
Train the model using the following script:
```bash
python scripts/model_training.py
```

### 6. Evaluate the Model
Evaluate the trained model with:
```bash
python scripts/model_evaluation.py
```

### 7. Jupyter Notebooks
For a more interactive experience, you can explore the project using the provided Jupyter Notebooks:

- 01_explore_data.ipynb: For data exploration and initial checks.

- 02_data_preprocessing.ipynb: For data cleaning and preprocessing.

- 03_model_training.ipynb: For training the model.

- 04_evaluation.ipynb: For evaluating the model’s performance.

#### Requirements:
 - pandas
 - numpy
 - scikit-learn
 - matplotlib
 - seaborn
 - joblib
 - kaggle

You can install all dependencies by running pip install -r requirements.txt.

## Conclusion
This project demonstrates the end-to-end pipeline for predicting credit risk. It covers data exploration, preprocessing, feature engineering, model training, and evaluation.

Feel free to explore the code, improve it, or deploy it for real-world use!

- yaml
- Copy
- Edit
