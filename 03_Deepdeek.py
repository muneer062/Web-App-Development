import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from io import StringIO, BytesIO

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                           accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for label encoding that preserves column names
class ColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.columns:
            if X_copy[col].dtype == 'object' or X_copy[col].dtype.name == 'category':
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    self.encoders[col].fit(X_copy[col])
                X_copy[col] = self.encoders[col].transform(X_copy[col])
        return X_copy
    
    def inverse_transform(self, X):
        X_copy = X.copy()
        for col in self.encoders:
            if col in X_copy.columns:
                X_copy[col] = self.encoders[col].inverse_transform(X_copy[col])
        return X_copy

# Set page config
st.set_page_config(page_title="ML App", layout="wide")

# Welcome message
st.title("🤖 Machine Learning Application")
st.write("""
Welcome to the Machine Learning Application!
This app allows you to perform both regression and classification tasks with your data.
Upload your dataset or use one of our example datasets to get started.
""")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

# Sidebar for data selection
st.sidebar.title("📊 Data Selection")
data_option = st.sidebar.radio(
    "Choose data source:",
    ("Upload your own data", "Use example data")
)

if data_option == "Upload your own data":
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset (CSV, TSV, Excel)",
        type=["csv", "tsv", "xlsx"]
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.sidebar.success("Dataset loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            
else:
    example_dataset = st.sidebar.selectbox(
        "Select example dataset",
        ["titanic", "tips", "iris"],
        index=0
    )
    
    try:
        df = sns.load_dataset(example_dataset)
        st.session_state.df = df
        st.sidebar.success(f"Loaded {example_dataset} dataset successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading example dataset: {e}")

# Display dataset information
if st.session_state.df is not None:
    df = st.session_state.df
    
    st.header("🔍 Dataset Information")
    
    # Show basic info in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["First Rows", "Shape", "Description", "Info"])
    
    with tab1:
        st.subheader("First 5 rows")
        st.dataframe(df.head())
        
    with tab2:
        st.subheader("Dataset Shape")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
    
    with tab3:
        st.subheader("Dataset Description")
        st.dataframe(df.describe())
        
    with tab4:
        st.subheader("Columns Information")
        buffer = StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
    
    # Feature and target selection
    st.header("🎯 Feature and Target Selection")
    
    all_columns = df.columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        features = st.multiselect(
            "Select features (independent variables)",
            all_columns,
            default=all_columns[:-1] if len(all_columns) > 1 else []
        )
    
    with col2:
        target = st.selectbox(
            "Select target (dependent variable)",
            [col for col in all_columns if col not in features] if features else all_columns
        )
    
    if features and target:
        # Determine problem type
        if pd.api.types.is_numeric_dtype(df[target]) and len(df[target].unique()) > 10:
            st.session_state.problem_type = "regression"
            st.success("✅ This appears to be a regression problem (continuous numeric target)")
        else:
            st.session_state.problem_type = "classification"
            st.success("✅ This appears to be a classification problem (categorical target)")
        
        # Data preprocessing
        st.header("⚙️ Data Preprocessing")
        
        # Separate features and target
        X = df[features]
        y = df[target]
        
        # Initialize preprocessing objects
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', IterativeImputer(max_iter=10, random_state=42)),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('label_encoder', ColumnLabelEncoder()),
            ('imputer', IterativeImputer(max_iter=10, random_state=42))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        st.session_state.preprocessor = preprocessor
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            st.warning(f"⚠️ Dataset contains {X.isnull().sum().sum()} missing values. They will be imputed.")
        
        # Show preprocessing summary
        with st.expander("Show preprocessing details"):
            st.write("Numeric features:", list(numeric_features))
            st.write("Categorical features:", list(categorical_features))
            st.write("Preprocessing steps:")
            st.write(numeric_transformer)
            st.write(categorical_transformer)
        
        # Train-test split
        st.header("✂️ Train-Test Split")
        test_size = st.slider(
            "Select test size ratio",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        st.write(f"Training set size: {X_train.shape[0]} samples")
        st.write(f"Test set size: {X_test.shape[0]} samples")
        
        # Model selection
        st.sidebar.header("🤖 Model Selection")
        
        if st.session_state.problem_type == "regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Support Vector Machine": SVR()
            }
            default_models = ["Linear Regression", "Random Forest"]
        else:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Support Vector Machine": SVC(probability=True, random_state=42)
            }
            default_models = ["Logistic Regression", "Random Forest"]
        
        selected_models = st.sidebar.multiselect(
            "Select models to train",
            list(models.keys()),
            default=default_models
        )
        
        # Train and evaluate models
        if st.button("🚀 Train Models"):
            st.header("📊 Model Training and Evaluation")
            
            best_score = -np.inf if st.session_state.problem_type == "regression" else 0
            st.session_state.best_model = None
            st.session_state.metrics = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, model_name in enumerate(selected_models):
                progress = (i + 1) / len(selected_models)
                progress_bar.progress(progress)
                status_text.text(f"Training {model_name}... ({int(progress*100)}%)")
                
                st.subheader(f"🧠 {model_name}")
                
                # Create pipeline
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', models[model_name])
                ])
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if st.session_state.problem_type == "classification" else None
                
                # Calculate metrics
                if st.session_state.problem_type == "regression":
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean Squared Error", f"{mse:.4f}")
                    col2.metric("Root Mean Squared Error", f"{rmse:.4f}")
                    col3.metric("Mean Absolute Error", f"{mae:.4f}")
                    col4.metric("R2 Score", f"{r2:.4f}")
                    
                    # ROC curve for regression (not typical, but included as requested)
                    fig, ax = plt.subplots()
                    fpr, tpr, _ = roc_curve(y_test > y_test.median(), y_pred)
                    roc_auc = roc_auc_score(y_test > y_test.median(), y_pred)
                    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                    
                    # Store metrics
                    st.session_state.metrics[model_name] = {
                        'MSE': mse,
                        'RMSE': rmse,
                        'MAE': mae,
                        'R2': r2,
                        'ROC AUC': roc_auc
                    }
                    
                    # Check if current model is the best
                    if r2 > best_score:
                        best_score = r2
                        st.session_state.best_model = pipeline
                
                else:  # classification
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) == 2 else None
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.4f}")
                    col2.metric("Precision", f"{precision:.4f}")
                    col3.metric("Recall", f"{recall:.4f}")
                    col4.metric("F1 Score", f"{f1:.4f}")
                    
                    if roc_auc is not None:
                        st.metric("ROC AUC Score", f"{roc_auc:.4f}")
                    
                        # ROC curve
                        fig, ax = plt.subplots()
                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                        ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                        ax.plot([0, 1], [0, 1], 'k--')
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('ROC Curve')
                        ax.legend(loc="lower right")
                        st.pyplot(fig)
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                    
                    # Store metrics
                    st.session_state.metrics[model_name] = {
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1': f1
                    }
                    if roc_auc is not None:
                        st.session_state.metrics[model_name]['ROC AUC'] = roc_auc
                    
                    # Check if current model is the best
                    if accuracy > best_score:
                        best_score = accuracy
                        st.session_state.best_model = pipeline
            
            progress_bar.empty()
            status_text.empty()
            
            # Display best model
            if st.session_state.best_model is not None:
                st.header("🏆 Best Model")
                
                if st.session_state.problem_type == "regression":
                    best_model_name = [name for name in selected_models 
                                     if st.session_state.metrics[name]['R2'] == best_score][0]
                else:
                    best_model_name = [name for name in selected_models 
                                     if st.session_state.metrics[name]['Accuracy'] == best_score][0]
                
                st.success(f"The best model is: {best_model_name}")
                
                # Display metrics comparison
                st.subheader("📈 Metrics Comparison")
                metrics_df = pd.DataFrame(st.session_state.metrics).T
                st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))
                
                # Download model
                st.header("💾 Model Download")
                if st.button("Download Best Model"):
                    model_bytes = pickle.dumps(st.session_state.best_model)
                    st.download_button(
                        label="Click to Download Model (Pickle)",
                        data=BytesIO(model_bytes),
                        file_name="best_model.pkl",
                        mime="application/octet-stream"
                    )
                
                # Prediction section
                st.header("🔮 Make Predictions")
                prediction_option = st.radio(
                    "Select prediction input method:",
                    ("Manual Input", "Upload File")
                )
                
                if prediction_option == "Manual Input":
                    st.subheader("Manual Input Parameters")
                    input_data = {}
                    
                    cols = st.columns(2)
                    col_idx = 0
                    
                    for feature in features:
                        with cols[col_idx]:
                            if pd.api.types.is_numeric_dtype(df[feature]):
                                min_val = float(df[feature].min())
                                max_val = float(df[feature].max())
                                default_val = float(df[feature].median())
                                input_data[feature] = st.slider(
                                    f"Select value for {feature}",
                                    min_val, max_val, default_val
                                )
                            else:
                                unique_vals = df[feature].unique().tolist()
                                input_data[feature] = st.selectbox(
                                    f"Select value for {feature}",
                                    unique_vals
                                )
                        col_idx = (col_idx + 1) % 2
                    
                    if st.button("Predict"):
                        input_df = pd.DataFrame([input_data])
                        prediction = st.session_state.best_model.predict(input_df)
                        
                        st.subheader("🎯 Prediction Result")
                        if st.session_state.problem_type == "regression":
                            st.success(f"Predicted value: {prediction[0]:.4f}")
                        else:
                            st.success(f"Predicted class: {prediction[0]}")
                
                else:  # Upload File
                    st.subheader("Upload File for Prediction")
                    pred_file = st.file_uploader(
                        "Upload file with data for prediction (CSV, TSV, Excel)",
                        type=["csv", "tsv", "xlsx"],
                        key="pred_file"
                    )
                    
                    if pred_file is not None:
                        try:
                            if pred_file.name.endswith('.xlsx'):
                                pred_df = pd.read_excel(pred_file)
                            else:
                                pred_df = pd.read_csv(pred_file)
                            
                            # Check if all features are present
                            missing_features = set(features) - set(pred_df.columns)
                            if missing_features:
                                st.error(f"Missing features in uploaded file: {', '.join(missing_features)}")
                            else:
                                pred_df = pred_df[features]
                                predictions = st.session_state.best_model.predict(pred_df)
                                
                                st.subheader("📊 Prediction Results")
                                result_df = pred_df.copy()
                                result_df['Prediction'] = predictions
                                st.dataframe(result_df)
                                
                                # Download predictions
                                csv = result_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "📥 Download Predictions as CSV",
                                    csv,
                                    "predictions.csv",
                                    "text/csv",
                                    key='download_csv'
                                )
                                
                        except Exception as e:
                            st.error(f"Error processing prediction file: {e}")
    else:
        st.warning("Please select both features and target variable to proceed.")
else:
    st.info("ℹ️ Please upload a dataset or select an example dataset to get started.")