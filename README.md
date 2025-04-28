# Credit Risk

This repository contains a **MATLAB** implementation for building a **credit scoring model** to predict loan status (good or bad) based on features from a dataset. The steps include data preprocessing, balancing the dataset using **SMOTE**, and training a **credit scorecard**. The performance is evaluated using various metrics such as accuracy, precision, recall, specificity, and F1 score.

## Key Steps:
1. **Data Preprocessing:**
   - Load and clean the dataset (`credit_dataset.csv`).
   - Convert the loan status to binary (0 for good loans, 1 for bad loans).
   - Remove unnecessary columns.

2. **SMOTE Balancing:**
   - Apply **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset, ensuring equal representation of good and bad loans.

3. **Credit Scorecard Model:**
   - Train a **credit scorecard** using the balanced dataset.
   - Generate **credit scores** for each record.
   - Visualize the **score distribution** by loan type.

4. **Performance Evaluation:**
   - Calculate and display **accuracy**, **precision**, **recall**, **specificity**, and **F1 score**.
   - Visualize the **confusion matrix** to compare true vs. predicted loan status.

## Requirements:
- MATLAB (or compatible environment for running the script)
- Necessary toolboxes for running the `creditscorecard` functions

## Usage:
1. Clone the repository and place your dataset (`credit_dataset.csv`) in the project directory.
2. Run the script to load and preprocess the data.
3. The model will train, balance the dataset, and produce a credit scorecard.
4. Results and visualizations, including performance metrics and confusion matrix, will be displayed.

## Metrics Interpretation:
- **Accuracy**: Measures how many predictions were correct (98.53% in the example).
- **Precision**: The ability to correctly identify bad loans (85.55%).
- **Recall**: The ability to correctly detect bad loans (91.49%).
- **Specificity**: The ability to correctly identify good loans (98.99%).
- **F1 Score**: The balance between precision and recall (88.42%).
