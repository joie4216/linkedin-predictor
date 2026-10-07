import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("Who Uses LinkedIn?")
st.write("Enter a person's information to predict whether they use LinkedIn.")


# ---- Data cleaning (same as Part 1) ----
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

s = pd.read_csv("social_media_usage.csv")

ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]),
    "income": np.where(s["income"] > 9, np.nan, s["income"]),
    "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent": np.where(s["par"] == 1, 1, 0),
    "married": np.where(s["marital"] == 1, 1, 0),
    "female": np.where(s["gender"] == 2, 1, 0),
    "age": np.where(s["age"] > 98, np.nan, s["age"])
})
ss = ss.dropna()


# ---- Model (same as Part 1) ----
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

lr = LogisticRegression(class_weight="balanced")
lr.fit(X_train, y_train)


# ---- User inputs ----
st.sidebar.header("Person's information")

income_options = {
    "Less than $10k": 1, "$10k-$20k": 2, "$20k-$30k": 3, "$30k-$40k": 4,
    "$40k-$50k": 5, "$50k-$75k": 6, "$75k-$100k": 7, "$100k-$150k": 8, "$150k+": 9
}
educ_options = {
    "Less than high school": 1, "High school incomplete": 2, "High school graduate": 3,
    "Some college": 4, "Associate degree": 5, "Bachelor's degree": 6,
    "Some postgraduate": 7, "Postgraduate degree": 8
}

income_label = st.sidebar.selectbox("Income", list(income_options.keys()), index=7)
educ_label = st.sidebar.selectbox("Education", list(educ_options.keys()), index=6)
age = st.sidebar.slider("Age", 18, 97, 42)
gender = st.sidebar.radio("Gender", ["Female", "Male"])
married = st.sidebar.radio("Married?", ["Yes", "No"])
parent = st.sidebar.radio("Parent of a child under 18?", ["No", "Yes"])

person = pd.DataFrame({
    "income": [income_options[income_label]],
    "education": [educ_options[educ_label]],
    "parent": [1 if parent == "Yes" else 0],
    "married": [1 if married == "Yes" else 0],
    "female": [1 if gender == "Female" else 0],
    "age": [age]
})


# ---- Prediction ----
prob = lr.predict_proba(person)[0, 1]
pred = lr.predict(person)[0]

st.header("Prediction")
if pred == 1:
    st.success("This person is classified as a LinkedIn user")
else:
    st.error("This person is classified as NOT a LinkedIn user")

st.metric("Probability of using LinkedIn", f"{prob:.1%}")
st.progress(float(prob))


# ---- How probability changes with age ----
st.header("How does age change the probability?")
st.write("Same person, only age changes:")

ages = list(range(18, 98))
age_df = pd.concat([person] * len(ages), ignore_index=True)
age_df["age"] = ages
age_df["probability"] = lr.predict_proba(age_df[X.columns])[:, 1]

st.line_chart(age_df.set_index("age")["probability"])


# ---- Segment analysis for marketing team ----
st.header("LinkedIn use by group (survey data)")

group = st.selectbox("Compare by", ["income", "education", "parent", "married", "female"])
rate = ss.groupby(group)["sm_li"].mean()
st.bar_chart(rate)
st.write(f"Overall LinkedIn use rate: {ss['sm_li'].mean():.1%}")


# ---- Model performance ----
st.header("Model performance")
y_pred = lr.predict(X_test)
st.write(f"Accuracy on test data: {accuracy_score(y_test, y_pred):.1%}")

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual: Not a user", "Actual: User"],
    columns=["Predicted: Not a user", "Predicted: User"]
)
st.dataframe(cm_df)
