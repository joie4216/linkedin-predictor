import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

st.set_page_config(page_title="Who Uses LinkedIn?", page_icon="💼", layout="wide")

st.title("Who Uses LinkedIn?")
st.write(
    "Enter a person's information in the sidebar to predict whether they use LinkedIn. "
    "The model is a logistic regression trained on Pew Research survey data on US social media use."
)


# ---- Data cleaning (same as Part 1) ----
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x


@st.cache_data
def load_data():
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
    return ss.dropna()


# ---- Model (same as Part 1) ----
@st.cache_resource
def train_model(ss):
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
    return lr, X.columns, X_test, y_test


ss = load_data()
lr, feature_cols, X_test, y_test = train_model(ss)


# ---- Readable labels ----
income_options = {
    "Less than $10k": 1, "$10k-$20k": 2, "$20k-$30k": 3, "$30k-$40k": 4,
    "$40k-$50k": 5, "$50k-$75k": 6, "$75k-$100k": 7, "$100k-$150k": 8, "$150k+": 9
}
educ_options = {
    "Less than high school": 1, "High school incomplete": 2, "High school graduate": 3,
    "Some college": 4, "Associate degree": 5, "Bachelor's degree": 6,
    "Some postgraduate": 7, "Postgraduate degree": 8
}


# ---- User inputs ----
st.sidebar.header("Person's information")

income_label = st.sidebar.selectbox("Income", list(income_options.keys()), index=7)
educ_label = st.sidebar.selectbox("Education", list(educ_options.keys()), index=6)
age = st.sidebar.slider("Age", 18, 97, 42)
gender = st.sidebar.radio("Gender", ["Female", "Male / other"])
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
prob = lr.predict_proba(person[feature_cols])[0, 1]
pred = lr.predict(person[feature_cols])[0]

st.header("Prediction")
col1, col2 = st.columns(2)
with col1:
    if pred == 1:
        st.success("This person is classified as a **LinkedIn user**")
    else:
        st.error("This person is classified as **NOT a LinkedIn user**")
with col2:
    st.metric("Probability of using LinkedIn", f"{prob:.1%}")
st.progress(float(prob))
st.caption(
    "People with a probability of 50% or higher are classified as users. "
    "Because the model uses balanced class weights, probabilities run higher than "
    "real-world rates (about 33% of respondents use LinkedIn), so they are best used "
    "to rank and compare people rather than as exact chances."
)


# ---- How probability changes with age ----
st.header("How does age change the probability?")
st.write("Same person, only age changes. The red dot marks the current age.")

ages = list(range(18, 98))
age_df = pd.concat([person] * len(ages), ignore_index=True)
age_df["age"] = ages
age_df["probability"] = lr.predict_proba(age_df[feature_cols])[:, 1]

line = alt.Chart(age_df).mark_line(strokeWidth=3).encode(
    x=alt.X("age:Q", title="Age"),
    y=alt.Y("probability:Q", title="Probability of using LinkedIn",
            axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
    tooltip=[alt.Tooltip("age:Q", title="Age"),
             alt.Tooltip("probability:Q", title="Probability", format=".1%")]
)
cutoff = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
    strokeDash=[6, 4], color="gray"
).encode(y="y:Q")
current = alt.Chart(pd.DataFrame({"age": [age], "probability": [prob]})).mark_circle(
    size=150, color="red"
).encode(x="age:Q", y="probability:Q")
label = alt.Chart(pd.DataFrame({"age": [age], "probability": [prob],
                                "text": [f"Age {age}: {prob:.1%}"]})).mark_text(
    align="left", dx=10, dy=-12, fontSize=13
).encode(x="age:Q", y="probability:Q", text="text:N")

st.altair_chart(line + cutoff + current + label, width="stretch")
st.caption("Dashed line = 50% classification cutoff.")


# ---- Segment analysis for marketing team ----
st.header("LinkedIn use by group (survey data)")
st.write("Which customer segments use LinkedIn most? These are actual usage rates from the survey.")

seg_df = ss.copy()
seg_df["Income"] = seg_df["income"].map({v: k for k, v in income_options.items()})
seg_df["Education"] = seg_df["education"].map({v: k for k, v in educ_options.items()})
seg_df["Age group"] = pd.cut(seg_df["age"], bins=[18, 30, 40, 50, 60, 70, 99],
                             labels=["18-29", "30-39", "40-49", "50-59", "60-69", "70+"],
                             right=False).astype(str)
seg_df["Parent"] = seg_df["parent"].map({0: "Not a parent", 1: "Parent"})
seg_df["Married"] = seg_df["married"].map({0: "Not married", 1: "Married"})
seg_df["Gender"] = seg_df["female"].map({0: "Male / other", 1: "Female"})

orders = {
    "Income": list(income_options.keys()),
    "Education": list(educ_options.keys()),
    "Age group": ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    "Parent": ["Not a parent", "Parent"],
    "Married": ["Not married", "Married"],
    "Gender": ["Male / other", "Female"],
}

group = st.selectbox("Compare by", list(orders.keys()))
rate = (seg_df.groupby(group)["sm_li"]
        .agg(rate="mean", respondents="size")
        .reindex(orders[group])
        .dropna()
        .reset_index())
overall = ss["sm_li"].mean()

bars = alt.Chart(rate).mark_bar(color="#4C72B0").encode(
    x=alt.X(f"{group}:N", sort=orders[group], title=group, axis=alt.Axis(labelAngle=-30)),
    y=alt.Y("rate:Q", title="Share using LinkedIn", axis=alt.Axis(format="%")),
    tooltip=[alt.Tooltip(f"{group}:N"),
             alt.Tooltip("rate:Q", title="Share using LinkedIn", format=".1%"),
             alt.Tooltip("respondents:Q", title="Respondents")]
)
avg = alt.Chart(pd.DataFrame({"y": [overall]})).mark_rule(
    strokeDash=[6, 4], color="red"
).encode(y="y:Q")

st.altair_chart(bars + avg, width="stretch")
st.caption(f"Red dashed line = overall LinkedIn use rate ({overall:.1%}).")


# ---- Model performance ----
st.header("Model performance")
y_pred = lr.predict(X_test)

m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
m2.metric("Precision", f"{precision_score(y_test, y_pred):.1%}")
m3.metric("Recall", f"{recall_score(y_test, y_pred):.1%}")

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual: Not a user", "Actual: User"],
    columns=["Predicted: Not a user", "Predicted: User"]
)
st.dataframe(cm_df)
st.caption(
    "Evaluated on 252 test respondents the model never saw during training. "
    "The model finds about 70% of actual LinkedIn users (recall), but fewer than half of the "
    "people it flags are actual users (precision), so it is best suited to low-cost, "
    "high-reach campaigns."
)
