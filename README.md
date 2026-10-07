# LinkedIn Predictor

This is our Streamlit app for the final project. It predicts whether someone uses LinkedIn based on a few basic things about them: income, education, age, gender, whether they're married, and whether they have kids.

Try it here: https://linkedin-predictor-ycahqhm3rvzygtgwk7swff.streamlit.app/

## What it does

You pick a person's info in the sidebar and the app tells you:
- if they'd be classified as a LinkedIn user or not
- the probability that they use LinkedIn

There's also a chart showing how the probability changes with age, a chart comparing LinkedIn use across different groups, and the model's accuracy and confusion matrix at the bottom.

## How it works

The model is the same logistic regression we built in Part 1. We used the Pew social media survey data, cleaned it (dropped missing values for income, education and age), split it 80/20 into train and test, and set class_weight to balanced. It gets about 65% accuracy on the test set.

## Files

- app.py - the Streamlit app
- requirements.txt - packages needed to run it
- social_media_usage.csv - the survey data (Pew, for class use only)

## Run it on your own computer
```
pip install -r requirements.txt
streamlit run app.py
```
