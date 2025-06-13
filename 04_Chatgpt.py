# Note: This code requires Streamlit and related libraries to run in an appropriate environment.
# It will not work in environments where external libraries like 'streamlit' are not available.
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import io
import matplotlib.pyplot as plt
import pickle

from sklearn.experimental import enable_iterative_imputer  # Needed to enable IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

# 1. Welcome message
st.title("🧠 Machine Learning Web App")
st.markdown("""
Welcome to the interactive ML Web Application! 🎉
Upload your dataset or choose from sample datasets, preprocess data, train models, evaluate performance, and make predictions.
""")

# 2. Upload or example data
data_choice = st.radio("Would you like to upload your data or use example data?", ("Upload Data", "Use Example Data"))

# 3 & 4. Handle file upload or default dataset
df = None
if data_choice == "Upload Data":
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "tsv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".tsv"):
                df = pd.read_csv(uploaded_file, sep='\t')
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading data: {e}")
else:
    dataset_name = st.sidebar.selectbox("Select a dataset", ["titanic", "tips", "iris"])
    df = sns.load_dataset(dataset_name)

# 5. Show basic data info
if df is not None:
    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Description:", df.describe())
    st.write("Columns:", df.columns.tolist())

    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text("Info:\n" + s)

    # 6. Column selection
    feature_cols = st.multiselect("Select feature columns", df.columns.tolist())
    target_col = st.selectbox("Select target column", df.columns.tolist())

    if feature_cols and target_col:
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # 7. Problem Type
        if pd.api.types.is_numeric_dtype(y):
            problem_type = "regression"
            st.success("Regression Problem Detected")
        else:
            problem_type = "classification"
            st.success("Classification Problem Detected")

        # 8. Preprocessing
        st.subheader("Preprocessing")
        encoders = {}

        # Fill missing values
        imputer = IterativeImputer()
        if X.isnull().sum().sum() > 0:
            X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
            st.info("Missing values filled with Iterative Imputer")

        # Label Encoding
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

        # Scaling
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

        # 9. Train-test split
        test_size = st.slider("Select test size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

        # 10. Model dictionary for all models
        if problem_type == "regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "SVM": SVR()
            }
        else:
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC(probability=True)
            }

        # 11–13. Train all models and evaluate
        st.subheader("Model Evaluation")
        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if problem_type == "regression":
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
            else:
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}

        results_df = pd.DataFrame(results).T
        st.dataframe(results_df.style.highlight_max(axis=0))

        # 14. Best model selection
        best_model_name = results_df.mean(axis=1).idxmax()
        best_model = models[best_model_name]
        st.success(f"Best Model Selected: {best_model_name}")

        # 15. Download model
        if st.checkbox("Download Trained Model"):
            with open("best_model.pkl", "wb") as f:
                pickle.dump(best_model, f)
            with open("best_model.pkl", "rb") as f:
                st.download_button("Download Model", f, file_name="best_model.pkl")

        # 16 & 17. Make Prediction
        if st.checkbox("Make Prediction"):
            prediction_mode = st.radio("Select input method", ["Manual Input", "Upload File"])
            input_df = None

            if prediction_mode == "Manual Input":
                input_data = []
                for col in feature_cols:
                    val = st.number_input(f"Input value for {col}")
                    input_data.append(val)
                input_df = pd.DataFrame([input_data], columns=feature_cols)

            else:
                input_file = st.file_uploader("Upload file for prediction", type=["csv"])
                if input_file is not None:
                    input_df = pd.read_csv(input_file)

            if input_df is not None:
                for col in input_df.select_dtypes(include='object').columns:
                    if col in encoders:
                        input_df[col] = encoders[col].transform(input_df[col])
                input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_cols)
                prediction = best_model.predict(input_df_scaled)
                st.write("Predictions:", prediction)