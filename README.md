# A Multifaceted Approach to Job Title Analysis
_CSE 519 - Data Science Fundamentals_

## Project Description
Project consists of three parts:
1. Salary Prediction
2. Job Clustering
3. Job Satisfaction Analysis

## Installing libraries
```
pip install -r requirements.txt
```

## File Descriptions
`Web Scraping Job titles.ipynb` - Code for Web Scraping Job titles from CareerBuilder.com  
`Salary Prediction.ipynb` - Code for Salary Prediction using Machine Learning  
`Job_Satisfaction.ipynb` - Code for Job Satisfaction Analysis and Graphs  
`run_app.py` - Code for running Streamlit app (Salary Prediction and Job Clustering)

## Datasets
`Job Information.csv` - Dataset built by scraping web data from CareerBuilder.com  
`WA_Fn-UseC_-HR-Employee-Attrition.csv` - Dataset download from Kaggle

## ML Model
`salary_model_30_11.pkl` - Weighted Model developed using a combination of Regressors (refer to Salary Prediction.ipynb)

## How to run the code
`.ipynb` files (Jupyter Notebook Files) can be run either using the command `jupyter notebook` or `jupyter lab`, or can be run directly on Google Colab (after mounting the Google Drive).

To run the file `run_app.py`, run the following command in the terminal:
```
streamlit run run_app.py
```

## Project Report
[A Multifaceted Approach to Job Title Analysis](/Job%20Title%20Analysis_Report.pdf)

