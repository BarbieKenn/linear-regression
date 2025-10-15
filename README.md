# Linear Regression from Scratch

A simple educational project where I implemented **linear regression** completely from scratch using only **NumPy** and **Pandas**.  
The goal was to understand the math and mechanics behind regression — not to rely on scikit-learn or other ML frameworks.

## Features

- Implemented both **analytic (normal equation)** and **gradient descent** methods  
- Used **California Housing dataset** from Kaggle  
- Added **batch gradient descent** support  
- Implemented core data preprocessing utilities:
  - Manual **one-hot encoding**
  - **Bias term** addition
  - **Normalization** and **standardization**
- Wrote three **evaluation metrics** from scratch

## Structure

Main modules include:
- Data preprocessing (`one_hot_encoding`, `add_bias`, `standardization`, `normalization`)
- Model training (`find_weights`, `gradient_descent_mse`)
- Evaluation (`metrics`)

## Requirements

All dependencies and their versions are listed in `requirements.txt`.

## Dataset

Dataset: [California Housing (Kaggle)](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

## Notes

This project is purely educational — focused on understanding the foundations of linear regression and numerical optimization.
