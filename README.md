# Banana Death Predictor

Computer Vision and Machine Learning Regression Project

---

## Overview

This project predicts the number of days remaining before a banana becomes completely spoiled using Computer Vision and Machine Learning.

The system analyzes banana images, extracts color and texture features using OpenCV, and predicts the remaining shelf life using a trained regression model.

Unlike normal classification projects that only label bananas as ripe or rotten, this project predicts an exact numerical value such as:

```text
5.3 days remaining
```

This makes the project more practical and technically advanced.

---

## Problem Statement

Food waste is a major global issue. Fruits are often discarded because it is difficult to estimate their remaining shelf life accurately.

This project demonstrates how Artificial Intelligence and Computer Vision can be used to:

* Predict fruit shelf life from images
* Reduce food waste
* Improve inventory management
* Support smart agriculture and retail systems

---

## How the System Works

```text
Banana Image
      ↓
OpenCV Feature Extraction
      ↓
Color + Texture Features
      ↓
Regression Model
      ↓
Predicted Days Remaining
```

---

## Features Extracted from Images

The system extracts multiple visual features from banana images:

| Feature           | Description                              |
| ----------------- | ---------------------------------------- |
| yellow_ratio      | Percentage of yellow pixels              |
| green_ratio       | Percentage of green pixels               |
| brown_ratio       | Percentage of brown pixels               |
| black_ratio       | Percentage of black pixels               |
| texture_roughness | Surface texture using Laplacian variance |
| mean_brightness   | Average brightness                       |
| std_brightness    | Brightness variation                     |
| edge_density      | Edge information using Canny detector    |
| hue_mean          | Mean HSV hue                             |
| saturation_mean   | Mean HSV saturation                      |

---

## Machine Learning Models Used

The following regression models were trained and evaluated:

| Model                       | MAE   | RMSE  | R² Score |
| --------------------------- | ----- | ----- | -------- |
| Ridge Regression            | 0.726 | 0.870 | 0.815    |
| Random Forest Regressor     | 0.648 | 0.868 | 0.815    |
| Gradient Boosting Regressor | 0.567 | 0.748 | 0.863    |

Gradient Boosting achieved the best performance and was selected as the final model.

---

## Technologies Used

* Python
* OpenCV
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Joblib

---

## Project Structure

```text
banana-death-predictor/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── best_model.pkl
│   └── feature_cols.pkl
│
├── src/
│   ├── feature_extraction.py
│   ├── train.py
│   └── predict.py
│
├── results/
│   ├── model_comparison.png
│   └── feature_importance.png
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shruti870/BANANA-DEATH-PREDICTOR.git
cd BANANA-DEATH-PREDICTOR
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run:

```bash
python src/train.py
```

This will:

* Extract image features
* Train regression models
* Compare performance
* Save the best model

---

## Running Predictions

Predict shelf life for a banana image:

```bash
python src/predict.py --image banana.jpg
```

Example Output:

```text
Prediction: 5.3 days remaining
Status: Eat within a few days
```

---

## Results

The system was tested on thousands of banana images across different ripeness stages.

| Banana Stage | Predicted Days |
| ------------ | -------------- |
| Unripe       | 12.0 days      |
| Ripe         | 5.3 days       |
| Overripe     | 2.9 days       |
| Rotten       | 0.0 days       |

---

## Future Improvements

* Build a Streamlit web application
* Use Deep Learning CNN models
* Extend to multiple fruits
* Deploy as a mobile application
* Improve prediction accuracy with larger datasets

---

## Dataset

Dataset Source:

https://www.kaggle.com/datasets/shahriar26s/banana-ripeness-classification-dataset



This project was developed as a Computer Vision and Regression project for machine learning portfolio development.
