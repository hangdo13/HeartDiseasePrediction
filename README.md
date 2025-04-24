# HeartDiseasePrediction
This project was developed as part of the IS7935 course and focuses on applying machine learning classification techniques to predict the presence of heart disease using a labeled dataset. The dataset, sourced from Kaggle, contains a variety of medical features (e.g., age, cholesterol levels, chest pain type, resting blood pressure) and a binary target indicating whether a patient has heart disease.

Objective
The primary goal is to build and evaluate multiple classification models to determine which performs best in predicting heart disease. The following models were implemented and tested:

Logistic Regression

Decision Tree

Random Forest

Majority Baseline Classifier (used as a performance benchmark)

Methodology
Using Python and Scikit-learn, the dataset was split into training and testing sets, and each model was trained on the training set and evaluated on the test set. Performance metrics used for evaluation included:

Accuracy – the overall correctness of predictions.

Precision – the proportion of positive predictions that were actually correct.

Recall – the ability of the model to correctly identify all actual positives.

F1 Score – the harmonic mean of precision and recall, balancing both metrics.

Each model was implemented with default parameters, and results were compared against the majority baseline classifier, which always predicts the most frequent class.

Key Takeaways and Analysis
This task is a binary classification problem, as the target variable contains two mutually exclusive outcomes: presence or absence of heart disease.

Compared to the majority baseline, all machine learning models performed significantly better, especially in recall and F1 score.

The majority baseline, while possibly achieving moderate accuracy due to class imbalance, is unsuitable for healthcare predictions, as it fails to detect true positive cases.

Recall was emphasized as the most important metric in this context. In health-related predictions, false negatives (failing to detect someone with heart disease) are more dangerous than false positives.

Among all models, Logistic Regression demonstrated the best overall performance, achieving the highest accuracy, precision, recall, and F1 score. This makes it the most reliable choice for this specific task.

However, it is important to consider F1 score alongside recall, as it provides a balanced view of the model's effectiveness, particularly when dealing with imbalanced datasets.
