import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Random Forest Regressor", layout="wide")

st.title("Random Forest Regressor")
st.write("Upload your dataset, train the model, and make predictions.")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------- COLUMN SELECTION ----------------
    st.subheader("Select Target Column")

    target_column = st.selectbox("Choose the target column", df.columns)

    # Features and Target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ---------------- HANDLE CATEGORICAL DATA ----------------
    X = pd.get_dummies(X, drop_first=True)

    # ---------------- TRAIN TEST SPLIT ----------------
    test_size = st.slider(
        "Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # ---------------- SCALING ----------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------- TRAIN MODEL ----------------
    if st.button("Train Model"):

        with st.spinner("Training Random Forest Regressor..."):

            model = RandomForestRegressor(random_state=42)

            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            }

            grid_search = GridSearchCV(
                estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1
            )

            grid_search.fit(X_train_scaled, y_train)

            best_model = grid_search.best_estimator_

            # Predictions
            y_pred = best_model.predict(X_test_scaled)

            # Metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            # ---------------- RESULTS ----------------
            st.success("Model Trained Successfully!")

            st.subheader("Best Hyperparameters")
            st.write(grid_search.best_params_)

            st.subheader("Model Performance")

            col1, col2, col3 = st.columns(3)

            col1.metric("R2 Score", round(r2, 4))
            col2.metric("MAE", round(mae, 4))
            col3.metric("MSE", round(mse, 4))


else:
    st.info("Please upload a CSV dataset to continue.")
