# Credit Card Fraud Detection

This project aims to detect credit card fraud.

## **Description**
This dataset contains credit card transactions made by European cardholders in the year 2023. It comprises over 550,000 records, and the data has been anonymized to protect the cardholders' identities. 
The primary objective of this dataset is to facilitate the development of fraud detection algorithms and models to identify potentially fraudulent transactions.

## **Key features of the data:**
*   `id:` Unique identifier for each transaction
*   `V1-V28:` Anonymized features representing various transaction attributes (e.g., time, location, etc.)
*   `Amount:` The transaction amount
*   `Class:` Binary label indicating whether the transaction is fraudulent (1) or not (0)

# Framework

## 1. Performance metrics
*   `Precision:` The proportion of correctly predicted positive cases out of all cases predicted as positive, indicating model reliability in positive predictions.
*   `Recall:` The proportion of actual positive cases that the model correctly identifies, measuring sensitivity to true positives.
*   `F1-score:` The harmonic mean of precision and recall, balancing both metrics to handle class imbalance effectively.
*   `AUC:` Measures the model’s overall classification performance by calculating the True Positive Rate (TPR) and False Positive Rate (FPR) across thresholds, then finding the area under the ROC curve.

## 2. Baseline model
I'll be using a baseline model as a refernce point to help me decide if the more complex models improve performance substantially.
The baseline model can also highlight data issues early. This will be used as the minimum performance benchmark, and will prevent unnecessary model complexity. For this I'll be using sklearn function,
but the other ones I'll build from scratch to maximize understanding and learning potential.

## 3. Autoencoder

## **Source:**
The dataset is from: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

You could use the Kaggle API for continouus updates, but it is just easier to do it manually. Therefore you could check if the data has been updated before using this. 


