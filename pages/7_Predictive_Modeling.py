import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import io
import json
import sys
import os
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import base64
import matplotlib.pyplot as plt

# Add the project root to the path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import get_data_summary, format_large_number

# Set page config
st.set_page_config(
    page_title="DataInsights Pro",
    layout="wide"
)

def main():
    # Page title
    st.title("🔮 Predictive Modeling")
    
    # Check if data exists in session state
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ Please upload a Dataset in the **Data Upload** page.")
        st.stop()

    # Get data from session state
    data = st.session_state.data
    
    # Sidebar for model selection
    model_type = st.sidebar.radio(
        "Choose Model Type",
        ["Classification", "Regression", "Clustering"]
    )
    
    # Main content
    if model_type == "Classification":
        classification_modeling(data)
    elif model_type == "Regression":
        regression_modeling(data)
    else:
        clustering_analysis(data)

def classification_modeling(data):
    st.header("🏷️ Classification Modeling")
    st.write("Classification models predict which category a data point belongs to.")
    
    # Check data columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    
    # Step 1: Select target variable
    st.subheader("1. Select Target Variable")
    
    # Find potential binary/categorical targets
    binary_targets = []
    for col in data.columns:
        n_unique = data[col].nunique()
        if n_unique >= 2 and n_unique <= 10:
            binary_targets.append(col)
    
    if not binary_targets:
        st.warning("No suitable target variables found. Classification requires a target with 2-10 unique values.")
        return
    
    target_col = st.selectbox(
        "Select target variable (2-10 classes):",
        binary_targets
    )
    
    # Display target distribution
    target_counts = data[target_col].value_counts()
    
    fig = px.pie(
        values=target_counts.values,
        names=target_counts.index,
        title=f"Distribution of {target_col}",
        hole=0.4
    )
    st.plotly_chart(fig, width=True)
    
    # Check if target is binary or multi-class
    n_classes = data[target_col].nunique()
    is_binary = n_classes == 2
    
    if is_binary:
        st.write(f"✅ Binary classification task with classes: {', '.join(map(str, data[target_col].unique()))}")
    else:
        st.write(f"✅ Multi-class classification task with {n_classes} classes: {', '.join(map(str, data[target_col].unique()))}")
    
    # Step 2: Select features
    st.subheader("2. Select Features")
    
    # Remove target from potential features
    potential_features = [col for col in data.columns if col != target_col]
    
    # Split into numeric and categorical for better organization
    numeric_features = [col for col in numeric_cols if col != target_col]
    categorical_features = [col for col in categorical_cols if col != target_col]
    
    st.write("Select numeric features:")
    selected_numeric = st.multiselect(
        "Numeric Features",
        numeric_features,
        default=numeric_features[:min(5, len(numeric_features))]
    )
    
    st.write("Select categorical features:")
    selected_categorical = st.multiselect(
        "Categorical Features",
        categorical_features,
        default=categorical_features[:min(3, len(categorical_features))]
    )
    
    # Combine selected features
    selected_features = selected_numeric + selected_categorical
    
    if not selected_features:
        st.warning("Please select at least one feature to proceed.")
        return
    
    # Step 3: Configure model
    st.subheader("3. Configure Model")
    
    # Define available classification models
    classification_models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Support Vector Machine": SVC(probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }
    
    # Select model
    selected_model = st.selectbox(
        "Select classification model:",
        list(classification_models.keys())
    )
    
    # Train-test split ratio
    test_size = st.slider(
        "Test set size (%):",
        min_value=10,
        max_value=50,
        value=20,
        step=5
    ) / 100
    
    # Advanced options
    show_advanced = st.checkbox("Show advanced options")
    
    if show_advanced:
        # Create tabs for different model types
        model_tabs = st.tabs(list(classification_models.keys()))
        
        # Logistic Regression parameters
        with model_tabs[0]:
            st.write("### Logistic Regression Parameters")
            lr_solver = st.selectbox(
                "Solver:",
                ["liblinear", "lbfgs", "newton-cg", "sag", "saga"],
                index=0
            )
            lr_c = st.number_input(
                "Regularization strength (C):",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.1
            )
            lr_max_iter = st.number_input(
                "Maximum iterations:",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )
            
            if selected_model == "Logistic Regression":
                model_params = {
                    'solver': lr_solver,
                    'C': lr_c,
                    'max_iter': lr_max_iter
                }
        
        # Decision Tree parameters
        with model_tabs[1]:
            st.write("### Decision Tree Parameters")
            dt_criterion = st.selectbox(
                "Criterion:",
                ["gini", "entropy"],
                index=0
            )
            dt_max_depth = st.number_input(
                "Maximum depth:",
                min_value=1,
                max_value=50,
                value=10,
                step=1
            )
            if dt_max_depth == 0:
                dt_max_depth = None
                
            dt_min_samples_split = st.number_input(
                "Minimum samples to split:",
                min_value=2,
                max_value=20,
                value=2,
                step=1
            )
            
            if selected_model == "Decision Tree":
                model_params = {
                    'criterion': dt_criterion,
                    'max_depth': dt_max_depth,
                    'min_samples_split': dt_min_samples_split
                }
        
        # Random Forest parameters
        with model_tabs[2]:
            st.write("### Random Forest Parameters")
            rf_n_estimators = st.number_input(
                "Number of trees:",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )
            rf_criterion = st.selectbox(
                "Criterion:",
                ["gini", "entropy"],
                index=0,
                key="rf_criterion"
            )
            rf_max_depth = st.number_input(
                "Maximum depth:",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                key="rf_max_depth"
            )
            if rf_max_depth == 0:
                rf_max_depth = None
            
            if selected_model == "Random Forest":
                model_params = {
                    'n_estimators': rf_n_estimators,
                    'criterion': rf_criterion,
                    'max_depth': rf_max_depth
                }
        
        # Gradient Boosting parameters
        with model_tabs[3]:
            st.write("### Gradient Boosting Parameters")
            gb_n_estimators = st.number_input(
                "Number of boosting stages:",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                key="gb_n_estimators"
            )
            gb_learning_rate = st.number_input(
                "Learning rate:",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01
            )
            gb_max_depth = st.number_input(
                "Maximum depth:",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                key="gb_max_depth"
            )
            
            if selected_model == "Gradient Boosting":
                model_params = {
                    'n_estimators': gb_n_estimators,
                    'learning_rate': gb_learning_rate,
                    'max_depth': gb_max_depth
                }
        
        # SVM parameters
        with model_tabs[4]:
            st.write("### Support Vector Machine Parameters")
            svm_kernel = st.selectbox(
                "Kernel:",
                ["linear", "poly", "rbf", "sigmoid"],
                index=2
            )
            svm_c = st.number_input(
                "Regularization parameter (C):",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key="svm_c"
            )
            svm_gamma = st.selectbox(
                "Kernel coefficient (gamma):",
                ["scale", "auto"],
                index=0
            )
            
            if selected_model == "Support Vector Machine":
                model_params = {
                    'kernel': svm_kernel,
                    'C': svm_c,
                    'gamma': svm_gamma,
                    'probability': True
                }
        
        # KNN parameters
        with model_tabs[5]:
            st.write("### K-Nearest Neighbors Parameters")
            knn_n_neighbors = st.number_input(
                "Number of neighbors:",
                min_value=1,
                max_value=20,
                value=5,
                step=1
            )
            knn_weights = st.selectbox(
                "Weight function:",
                ["uniform", "distance"],
                index=0
            )
            knn_metric = st.selectbox(
                "Distance metric:",
                ["euclidean", "manhattan", "minkowski"],
                index=0
            )
            
            if selected_model == "K-Nearest Neighbors":
                model_params = {
                    'n_neighbors': knn_n_neighbors,
                    'weights': knn_weights,
                    'metric': knn_metric
                }
    else:
        # Default parameters if advanced options are not shown
        model_params = {}
    
    # Step 4: Train and evaluate
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            # Prepare data
            X = data[selected_features].copy()
            y = data[target_col].copy()
            
            # Encode target if it's not numeric
            if not pd.api.types.is_numeric_dtype(y):
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                class_names = label_encoder.classes_
            else:
                class_names = sorted(y.unique())
            
            # Split data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Create preprocessing pipeline
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, selected_numeric),
                        ('cat', categorical_transformer, selected_categorical)
                    ]
                )
                
                # Create model pipeline
                model_instance = classification_models[selected_model]
                
                # Set model parameters if provided
                if model_params:
                    model_instance.set_params(**model_params)
                
                # Combine preprocessing and model
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', model_instance)
                ])
                
                # Fit model
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                
                # For ROC curve and probability-based metrics
                if hasattr(pipeline, 'predict_proba'):
                    y_prob = pipeline.predict_proba(X_test)
                else:
                    y_prob = None
                
                # Display results
                st.subheader("4. Model Results")
                
                # Classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Display metrics based on binary or multi-class
                if is_binary:
                    precision = precision_score(y_test, y_pred, average='binary')
                    recall = recall_score(y_test, y_pred, average='binary')
                    f1 = f1_score(y_test, y_pred, average='binary')
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.2%}")
                    col2.metric("Precision", f"{precision:.2%}")
                    col3.metric("Recall", f"{recall:.2%}")
                    col4.metric("F1 Score", f"{f1:.2%}")
                else:
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.2%}")
                    col2.metric("Precision (weighted)", f"{precision:.2%}")
                    col3.metric("Recall (weighted)", f"{recall:.2%}")
                    col4.metric("F1 Score (weighted)", f"{f1:.2%}")
                
                # Confusion Matrix
                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                
                # Create proper labels for confusion matrix
                if len(class_names) <= 10:  # Only show class names if not too many
                    labels = class_names
                else:
                    labels = [f"Class {i}" for i in range(len(class_names))]
                
                # Create heatmap
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=labels,
                    y=labels,
                    text_auto=True,
                    title="Confusion Matrix",
                    color_continuous_scale='Blues'
                )
                
                st.plotly_chart(fig, width=True)
                
                # ROC curve for binary classification
                if is_binary and y_prob is not None:
                    st.write("### ROC Curve")
                    
                    # Compute ROC curve and ROC area
                    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    # Create ROC curve
                    fig = px.area(
                        x=fpr, y=tpr,
                        title=f'ROC Curve (AUC = {roc_auc:.4f})',
                        labels=dict(x='False Positive Rate', y='True Positive Rate'),
                        width=700, height=500
                    )
                    
                    # Add diagonal reference line
                    fig.add_shape(
                        type='line',
                        line=dict(dash='dash'),
                        x0=0, x1=1, y0=0, y1=1
                    )
                    
                    st.plotly_chart(fig, width=True)
                
                # Feature importance for models that support it
                if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                    st.write("### Feature Importance")
                    
                    # Get feature names after preprocessing
                    if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
                        # For newer scikit-learn versions
                        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                    else:
                        # Fallback to indices
                        feature_names = [f"Feature {i}" for i in range(len(pipeline.named_steps['model'].feature_importances_))]
                    
                    # Create DataFrame for importances
                    importances = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': pipeline.named_steps['model'].feature_importances_
                    })
                    
                    # Sort by importance
                    importances = importances.sort_values('Importance', ascending=False)
                    
                    # Plot
                    fig = px.bar(
                        importances.head(15),  # Show top 15
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance',
                        labels={'Importance': 'Importance', 'Feature': 'Feature'}
                    )
                    
                    st.plotly_chart(fig, width=True)
                
                # Or coefficient values for linear models
                elif hasattr(pipeline.named_steps['model'], 'coef_'):
                    st.write("### Feature Coefficients")
                    
                    # Get feature names after preprocessing
                    if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
                        # For newer scikit-learn versions
                        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                    else:
                        # Fallback to indices
                        feature_names = [f"Feature {i}" for i in range(pipeline.named_steps['model'].coef_.shape[1] if len(pipeline.named_steps['model'].coef_.shape) > 1 else len(pipeline.named_steps['model'].coef_))]
                    
                    # Handle multi-class vs binary case
                    if len(pipeline.named_steps['model'].coef_.shape) > 1 and pipeline.named_steps['model'].coef_.shape[0] > 1:
                        # Multi-class case - just show first class for simplicity
                        coefs = pipeline.named_steps['model'].coef_[0]
                    else:
                        # Binary case
                        coefs = pipeline.named_steps['model'].coef_.ravel()
                    
                    # Create DataFrame for coefficients
                    coefficients = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': coefs
                    })
                    
                    # Sort by absolute value
                    coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
                    coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False).drop('Abs_Coefficient', axis=1)
                    
                    # Plot
                    fig = px.bar(
                        coefficients.head(15),  # Show top 15
                        x='Coefficient',
                        y='Feature',
                        orientation='h',
                        title='Feature Coefficients',
                        labels={'Coefficient': 'Coefficient', 'Feature': 'Feature'},
                        color='Coefficient',
                        color_continuous_scale='RdBu_r'
                    )
                    
                    st.plotly_chart(fig, width=True)
                
                # Cross-validation
                st.write("### Cross-Validation")
                
                cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
                
                st.write(f"5-fold cross-validation accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
                
                # Plot CV scores
                fig = px.bar(
                    x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                    y=cv_scores,
                    labels={'x': 'Fold', 'y': 'Accuracy'},
                    title='Cross-Validation Accuracy by Fold'
                )
                
                fig.add_hline(
                    y=cv_scores.mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {cv_scores.mean():.2%}",
                    annotation_position="bottom right"
                )
                
                st.plotly_chart(fig, width=True)
                
                # Save model option
                st.write("### Save Model")
                st.write("You can save this model to use it for predictions on new data.")
                
                # Save model in session state
                if 'models' not in st.session_state:
                    st.session_state['models'] = {}
                
                model_name = st.text_input("Enter a name for this model:", f"{selected_model} ({target_col})")
                
                if st.button("Save Model"):
                    # Store the model and related info
                    st.session_state['models'][model_name] = {
                        'pipeline': pipeline,
                        'target': target_col,
                        'features': selected_features,
                        'accuracy': accuracy,
                        'type': 'classification',
                        'class_names': class_names
                    }
                    
                    st.success(f"Model '{model_name}' saved successfully!")
                    
                    # Show list of saved models
                    if len(st.session_state['models']) > 0:
                        st.write("### Saved Models")
                        
                        for name, model_info in st.session_state['models'].items():
                            st.write(f"**{name}**: {model_info['type'].capitalize()} model for '{model_info['target']}' (Accuracy: {model_info['accuracy']:.2%})")
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.write("Common issues:")
                st.write("- Missing values in the data")
                st.write("- Too few samples for one of the classes")
                st.write("- High cardinality categorical features")
                st.write("- Perfect separation (for logistic regression)")
        
def regression_modeling(data):
    st.header("📈 Regression Modeling")
    st.write("Regression models predict continuous numeric values.")
    
    # Check data columns
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(numeric_cols) < 1:
        st.warning("No numeric columns found. Regression requires numeric target variables.")
        return
    
    # Step 1: Select target variable
    st.subheader("1. Select Target Variable")
    
    target_col = st.selectbox(
        "Select numeric target variable:",
        numeric_cols
    )
    
    # Display target distribution
    fig = px.histogram(
        data,
        x=target_col,
        title=f"Distribution of {target_col}",
        marginal="box"
    )
    st.plotly_chart(fig, width=True)
    
    # Step 2: Select features
    st.subheader("2. Select Features")
    
    # Remove target from potential features
    numeric_features = [col for col in numeric_cols if col != target_col]
    categorical_features = categorical_cols
    
    st.write("Select numeric features:")
    selected_numeric = st.multiselect(
        "Numeric Features",
        numeric_features,
        default=numeric_features[:min(5, len(numeric_features))]
    )
    
    st.write("Select categorical features:")
    selected_categorical = st.multiselect(
        "Categorical Features",
        categorical_features,
        default=categorical_features[:min(3, len(categorical_features))]
    )
    
    # Combine selected features
    selected_features = selected_numeric + selected_categorical
    
    if not selected_features:
        st.warning("Please select at least one feature to proceed.")
        return
    
    # Step 3: Configure model
    st.subheader("3. Configure Model")
    
    # Define available regression models
    regression_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Support Vector Regression": SVR(),
        "K-Nearest Neighbors": KNeighborsRegressor()
    }
    
    # Select model
    selected_model = st.selectbox(
        "Select regression model:",
        list(regression_models.keys())
    )
    
    # Train-test split ratio
    test_size = st.slider(
        "Test set size (%):",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        key="regression_test_size"
    ) / 100
    
    # Advanced options
    show_advanced = st.checkbox("Show advanced options", key="regression_advanced")
    
    if show_advanced:
        # Create tabs for different model types
        model_tabs = st.tabs(list(regression_models.keys()))
        
        # Linear Regression parameters
        with model_tabs[0]:
            st.write("### Linear Regression Parameters")
            st.write("Linear Regression has no hyperparameters to tune.")
            
            if selected_model == "Linear Regression":
                model_params = {}
        
        # Ridge Regression parameters
        with model_tabs[1]:
            st.write("### Ridge Regression Parameters")
            ridge_alpha = st.number_input(
                "Regularization strength (alpha):",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.1
            )
            
            if selected_model == "Ridge Regression":
                model_params = {
                    'alpha': ridge_alpha
                }
        
        # Lasso Regression parameters
        with model_tabs[2]:
            st.write("### Lasso Regression Parameters")
            lasso_alpha = st.number_input(
                "Regularization strength (alpha):",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key="lasso_alpha"
            )
            
            if selected_model == "Lasso Regression":
                model_params = {
                    'alpha': lasso_alpha
                }
        
        # Decision Tree parameters
        with model_tabs[3]:
            st.write("### Decision Tree Parameters")
            dt_criterion = st.selectbox(
                "Criterion:",
                ["mse", "friedman_mse", "mae"],
                index=0,
                key="dt_reg_criterion"
            )
            dt_max_depth = st.number_input(
                "Maximum depth:",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                key="dt_reg_max_depth"
            )
            if dt_max_depth == 0:
                dt_max_depth = None
                
            dt_min_samples_split = st.number_input(
                "Minimum samples to split:",
                min_value=2,
                max_value=20,
                value=2,
                step=1,
                key="dt_reg_min_samples_split"
            )
            
            if selected_model == "Decision Tree":
                model_params = {
                    'criterion': dt_criterion,
                    'max_depth': dt_max_depth,
                    'min_samples_split': dt_min_samples_split
                }
        
        # Random Forest parameters
        with model_tabs[4]:
            st.write("### Random Forest Parameters")
            rf_n_estimators = st.number_input(
                "Number of trees:",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                key="rf_reg_n_estimators"
            )
            rf_criterion = st.selectbox(
                "Criterion:",
                ["mse", "mae"],
                index=0,
                key="rf_reg_criterion"
            )
            rf_max_depth = st.number_input(
                "Maximum depth:",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                key="rf_reg_max_depth"
            )
            if rf_max_depth == 0:
                rf_max_depth = None
            
            if selected_model == "Random Forest":
                model_params = {
                    'n_estimators': rf_n_estimators,
                    'criterion': rf_criterion,
                    'max_depth': rf_max_depth
                }
        
        # Gradient Boosting parameters
        with model_tabs[5]:
            st.write("### Gradient Boosting Parameters")
            gb_n_estimators = st.number_input(
                "Number of boosting stages:",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                key="gb_reg_n_estimators"
            )
            gb_learning_rate = st.number_input(
                "Learning rate:",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01,
                key="gb_reg_learning_rate"
            )
            gb_max_depth = st.number_input(
                "Maximum depth:",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                key="gb_reg_max_depth"
            )
            
            if selected_model == "Gradient Boosting":
                model_params = {
                    'n_estimators': gb_n_estimators,
                    'learning_rate': gb_learning_rate,
                    'max_depth': gb_max_depth
                }
        
        # SVR parameters
        with model_tabs[6]:
            st.write("### Support Vector Regression Parameters")
            svr_kernel = st.selectbox(
                "Kernel:",
                ["linear", "poly", "rbf", "sigmoid"],
                index=2,
                key="svr_kernel"
            )
            svr_c = st.number_input(
                "Regularization parameter (C):",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key="svr_c"
            )
            svr_epsilon = st.number_input(
                "Epsilon:",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01
            )
            
            if selected_model == "Support Vector Regression":
                model_params = {
                    'kernel': svr_kernel,
                    'C': svr_c,
                    'epsilon': svr_epsilon
                }
        
        # KNN parameters
        with model_tabs[7]:
            st.write("### K-Nearest Neighbors Parameters")
            knn_n_neighbors = st.number_input(
                "Number of neighbors:",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="knn_reg_n_neighbors"
            )
            knn_weights = st.selectbox(
                "Weight function:",
                ["uniform", "distance"],
                index=0,
                key="knn_reg_weights"
            )
            knn_metric = st.selectbox(
                "Distance metric:",
                ["euclidean", "manhattan", "minkowski"],
                index=0,
                key="knn_reg_metric"
            )
            
            if selected_model == "K-Nearest Neighbors":
                model_params = {
                    'n_neighbors': knn_n_neighbors,
                    'weights': knn_weights,
                    'metric': knn_metric
                }
    else:
        # Default parameters if advanced options are not shown
        model_params = {}
    
    # Step 4: Train and evaluate
    if st.button("Train Model", key="train_regression"):
        with st.spinner("Training model..."):
            # Prepare data
            X = data[selected_features].copy()
            y = data[target_col].copy()
            
            # Split data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Create preprocessing pipeline
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, selected_numeric),
                        ('cat', categorical_transformer, selected_categorical)
                    ]
                )
                
                # Create model pipeline
                model_instance = regression_models[selected_model]
                
                # Set model parameters if provided
                if model_params:
                    model_instance.set_params(**model_params)
                
                # Combine preprocessing and model
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', model_instance)
                ])
                
                # Fit model
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                
                # Display results
                st.subheader("4. Model Results")
                
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("R² Score", f"{r2:.4f}")
                col2.metric("RMSE", f"{rmse:.4f}")
                col3.metric("MAE", f"{mae:.4f}")
                col4.metric("MSE", f"{mse:.4f}")
                
                # Predicted vs Actual plot
                st.write("### Predicted vs Actual Values")
                
                # Create scatter plot
                fig = px.scatter(
                    x=y_test,
                    y=y_pred,
                    labels={"x": "Actual", "y": "Predicted"},
                    title="Predicted vs Actual Values"
                )
                
                # Add 45-degree reference line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                padding = (max_val - min_val) * 0.1
                
                fig.add_trace(
                    go.Scatter(
                        x=[min_val - padding, max_val + padding],
                        y=[min_val - padding, max_val + padding],
                        mode='lines',
                        name='Perfect Predictions',
                        line=dict(dash='dash', color='gray')
                    )
                )
                
                st.plotly_chart(fig, width=True)
                
                # Residuals plot
                st.write("### Residual Plot")
                
                residuals = y_test - y_pred
                
                fig = px.scatter(
                    x=y_pred,
                    y=residuals,
                    labels={"x": "Predicted Values", "y": "Residuals"},
                    title="Residual Plot"
                )
                
                # Add horizontal reference line at y=0
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="red"
                )
                
                st.plotly_chart(fig, width=True)
                
                # Residuals distribution
                st.write("### Residuals Distribution")
                
                fig = px.histogram(
                    residuals,
                    labels={"value": "Residual", "count": "Frequency"},
                    title="Distribution of Residuals",
                    marginal="box"
                )
                
                st.plotly_chart(fig, width=True)
                
                # Feature importance for models that support it
                if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                    st.write("### Feature Importance")
                    
                    # Get feature names after preprocessing
                    if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
                        # For newer scikit-learn versions
                        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                    else:
                        # Fallback to indices
                        feature_names = [f"Feature {i}" for i in range(len(pipeline.named_steps['model'].feature_importances_))]
                    
                    # Create DataFrame for importances
                    importances = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': pipeline.named_steps['model'].feature_importances_
                    })
                    
                    # Sort by importance
                    importances = importances.sort_values('Importance', ascending=False)
                    
                    # Plot
                    fig = px.bar(
                        importances.head(15),  # Show top 15
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance',
                        labels={'Importance': 'Importance', 'Feature': 'Feature'}
                    )
                    
                    st.plotly_chart(fig, width=True)
                
                # Or coefficient values for linear models
                elif hasattr(pipeline.named_steps['model'], 'coef_'):
                    st.write("### Feature Coefficients")
                    
                    # Get feature names after preprocessing
                    if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
                        # For newer scikit-learn versions
                        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                    else:
                        # Fallback to indices
                        feature_names = [f"Feature {i}" for i in range(len(pipeline.named_steps['model'].coef_))]
                    
                    # Create DataFrame for coefficients
                    coefficients = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': pipeline.named_steps['model'].coef_
                    })
                    
                    # Add intercept
                    if hasattr(pipeline.named_steps['model'], 'intercept_'):
                        st.write(f"**Intercept:** {pipeline.named_steps['model'].intercept_:.4f}")
                    
                    # Sort by absolute value
                    coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
                    coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False).drop('Abs_Coefficient', axis=1)
                    
                    # Plot
                    fig = px.bar(
                        coefficients.head(15),  # Show top 15
                        x='Coefficient',
                        y='Feature',
                        orientation='h',
                        title='Feature Coefficients',
                        labels={'Coefficient': 'Coefficient', 'Feature': 'Feature'},
                        color='Coefficient',
                        color_continuous_scale='RdBu_r'
                    )
                    
                    st.plotly_chart(fig, width=True)
                
                # Cross-validation
                st.write("### Cross-Validation")
                
                cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
                
                st.write(f"5-fold cross-validation R² score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
                # Plot CV scores
                fig = px.bar(
                    x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                    y=cv_scores,
                    labels={'x': 'Fold', 'y': 'R² Score'},
                    title='Cross-Validation R² Score by Fold'
                )
                
                fig.add_hline(
                    y=cv_scores.mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {cv_scores.mean():.4f}",
                    annotation_position="bottom right"
                )
                
                st.plotly_chart(fig, width=True)
                
                # Save model option
                st.write("### Save Model")
                st.write("You can save this model to use it for predictions on new data.")
                
                # Save model in session state
                if 'models' not in st.session_state:
                    st.session_state['models'] = {}
                
                model_name = st.text_input("Enter a name for this model:", f"{selected_model} ({target_col})")
                
                if st.button("Save Model", key="save_reg_model"):
                    # Store the model and related info
                    st.session_state['models'][model_name] = {
                        'pipeline': pipeline,
                        'target': target_col,
                        'features': selected_features,
                        'r2': r2,
                        'type': 'regression'
                    }
                    
                    st.success(f"Model '{model_name}' saved successfully!")
                    
                    # Show list of saved models
                    if len(st.session_state['models']) > 0:
                        st.write("### Saved Models")
                        
                        for name, model_info in st.session_state['models'].items():
                            if model_info['type'] == 'regression':
                                st.write(f"**{name}**: Regression model for '{model_info['target']}' (R²: {model_info['r2']:.4f})")
                            else:
                                st.write(f"**{name}**: Classification model for '{model_info['target']}' (Accuracy: {model_info['accuracy']:.2%})")
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.write("Common issues:")
                st.write("- Missing values in the data")
                st.write("- High cardinality categorical features")
                st.write("- Perfect collinearity between features")

def clustering_analysis(data):
    st.header("Clustering Analysis")
    st.markdown("""
    Clustering algorithms group similar data points together based on their features, 
    helping you discover hidden patterns and structures in your data.
    """)
    
    # Check if there are numeric columns for clustering
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("⚠️ No numeric columns found in your dataset. Clustering requires numeric features.")
        return
    
    # Data preparation
    st.subheader("1. Data Preparation")
    
    # Feature selection
    feature_cols = st.multiselect(
        "Select numeric columns for clustering",
        options=numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))],
        key="clustering_features"
    )
    
    if not feature_cols:
        st.warning("⚠️ Please select at least one feature column to continue.")
        return
    
    # Handle missing values
    if data[feature_cols].isna().any().any():
        st.warning(f"⚠️ Selected features contain missing values which can affect clustering.")
        handle_missing = st.radio(
            "How would you like to handle missing values?",
            options=["Drop rows with missing values", "Fill missing values with mean", "Fill missing values with median"],
            index=0,
            key="clustering_missing"
        )
        
        if handle_missing == "Drop rows with missing values":
            clean_data = data.dropna(subset=feature_cols).copy()
            if len(clean_data) < len(data):
                st.info(f"Dropped {len(data) - len(clean_data)} rows with missing values ({(1 - len(clean_data)/len(data)):.1%} of data).")
        elif handle_missing == "Fill missing values with mean":
            clean_data = data.copy()
            for col in feature_cols:
                clean_data[col] = clean_data[col].fillna(clean_data[col].mean())
            st.info("Missing values filled with column means.")
        else:
            clean_data = data.copy()
            for col in feature_cols:
                clean_data[col] = clean_data[col].fillna(clean_data[col].median())
            st.info("Missing values filled with column medians.")
    else:
        clean_data = data.copy()
    
    # Standardize the data
    st.write("Data Standardization:")
    standardize = st.radio(
        "Standardize features?",
        options=["Yes, standardize (recommended)", "No, use raw values"],
        index=0,
        key="clustering_standardize"
    )
    
    # Choose clustering method
    st.subheader("2. Clustering Method")
    
    clustering_method = st.radio(
        "Select clustering algorithm",
        options=["K-Means Clustering", "Hierarchical Clustering", "DBSCAN Clustering"],
        index=0,
        key="clustering_method"
    )
    
    # Create a container for visualizations
    viz_container = st.container()
    
    # Display clustering results based on selected method
    if clustering_method == "K-Means Clustering":
        kmeans_clustering(clean_data, feature_cols, standardize, viz_container)
    elif clustering_method == "Hierarchical Clustering":
        hierarchical_clustering(clean_data, feature_cols, standardize, viz_container)
    else:  # DBSCAN
        dbscan_clustering(clean_data, feature_cols, standardize, viz_container)

def kmeans_clustering(data, numeric_cols, standardize, viz_container):
    # K-Means parameters
    st.write("K-Means Parameters:")
    
    k_value = st.slider(
        "Number of Clusters (k)",
        min_value=2,
        max_value=min(15, len(data) // 20),  # Limit based on data size
        value=3,
        key="kmeans_k"
    )
    
    random_state = 42
    
    # Find optimal k using elbow method
    if st.button("Find Optimal Number of Clusters", key="find_optimal_k"):
        with st.spinner("Finding optimal clusters using the Elbow Method..."):
            # Extract features
            X = data[numeric_cols].values
            
            # Standardize if selected
            if standardize == "Yes, standardize (recommended)":
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Calculate inertia for different k values
            inertias = []
            k_range = range(1, min(11, len(data) // 20 + 1))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point
            elbow_point = find_elbow_point(list(k_range), inertias)
            
            # Plot elbow curve
            with viz_container:
                st.subheader("Elbow Method for Optimal k")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(k_range), 
                    y=inertias,
                    mode='lines+markers',
                    name='Inertia'
                ))
                
                # Add elbow point marker
                fig.add_trace(go.Scatter(
                    x=[elbow_point],
                    y=[inertias[elbow_point-1]],
                    mode='markers',
                    marker=dict(size=12, color='red'),
                    name=f'Elbow Point (k={elbow_point})'
                ))
                
                fig.update_layout(
                    title="Elbow Method for Optimal Number of Clusters",
                    xaxis_title="Number of Clusters (k)",
                    yaxis_title="Inertia (Sum of Squared Distances)",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"The optimal number of clusters appears to be around **{elbow_point}**. "
                        f"This is where adding more clusters provides diminishing returns in terms of explained variance.")
    
    # Run K-Means clustering button
    if st.button("Run K-Means Clustering", key="run_kmeans"):
        with st.spinner("Running K-Means clustering and visualizing results..."):
            # Extract features
            X = data[numeric_cols].values
            
            # Standardize if selected
            if standardize == "Yes, standardize (recommended)":
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Fit KMeans
            kmeans = KMeans(n_clusters=k_value, random_state=random_state, n_init='auto')
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add clusters to dataframe
            cluster_data = data.copy()
            cluster_data['Cluster'] = clusters
            
            # Save clusters to session state
            if 'clustering_results' not in st.session_state:
                st.session_state.clustering_results = {}
            
            st.session_state.clustering_results['kmeans'] = {
                'data': cluster_data,
                'method': 'K-Means',
                'params': {'k': k_value},
                'columns': numeric_cols,
                'clusters': clusters,
                'model': kmeans,
                'scaler': scaler if standardize == "Yes, standardize (recommended)" else None
            }
            
            # Draw visualizations
            with viz_container:
                st.subheader("K-Means Clustering Results")
                
                # 1. Show number of points in each cluster
                st.write("#### Cluster Sizes")
                cluster_counts = pd.DataFrame(cluster_data['Cluster'].value_counts()).reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                cluster_counts = cluster_counts.sort_values('Cluster')
                
                fig = px.bar(
                    cluster_counts,
                    x='Cluster',
                    y='Count',
                    color='Cluster',
                    title='Number of Data Points per Cluster'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. Perform dimensionality reduction for visualization
                if len(numeric_cols) > 2:
                    st.write("#### 2D Cluster Visualization")
                    st.info("Using PCA to visualize high-dimensional clusters in 2D space.")
                    
                    # PCA for dimensionality reduction
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    # Create a DataFrame for plotting
                    pca_df = pd.DataFrame({
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1],
                        'Cluster': clusters
                    })
                    
                    # Plot clusters
                    fig = px.scatter(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        color='Cluster',
                        title='PCA Projection of Clusters',
                        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                        category_orders={'Cluster': sorted(pca_df['Cluster'].unique())},
                        color_discrete_sequence=px.colors.qualitative.G10
                    )
                    
                    # Add cluster centers
                    centers_pca = pca.transform(kmeans.cluster_centers_)
                    
                    for i, center in enumerate(centers_pca):
                        fig.add_trace(go.Scatter(
                            x=[center[0]],
                            y=[center[1]],
                            mode='markers',
                            marker=dict(
                                symbol='x',
                                size=20,
                                color='red',
                                line=dict(width=2)
                            ),
                            name=f'Centroid {i}',
                            showlegend=True
                        ))
                    
                    st.plotly_chart(fig)
                    
                    # Variance explained
                    var_explained = pca.explained_variance_ratio_
                    st.write(f"Variance explained: PC1 ({var_explained[0]:.2%}), PC2 ({var_explained[1]:.2%}), "
                             f"Total ({sum(var_explained):.2%})")
                    
                    # Show warning if low variance explained
                    if sum(var_explained) < 0.5:
                        st.warning("⚠️ Low variance explained by the first 2 principal components. "
                                   "This 2D visualization may not accurately represent the actual clusters in higher dimensions.")
                else:
                    # Direct scatter plot for 2 features
                    st.write("#### Cluster Visualization")
                    
                    if len(numeric_cols) == 2:
                        # Just plot the two columns
                        fig = px.scatter(
                            cluster_data,
                            x=numeric_cols[0],
                            y=numeric_cols[1],
                            color='Cluster',
                            title=f'Clusters by {numeric_cols[0]} and {numeric_cols[1]}',
                            category_orders={'Cluster': sorted(cluster_data['Cluster'].unique())},
                            color_discrete_sequence=px.colors.qualitative.G10
                        )
                        
                        # Add cluster centers (original scale)
                        for i, center in enumerate(kmeans.cluster_centers_):
                            # Transform center back to original scale if needed
                            if standardize == "Yes, standardize (recommended)":
                                original_center = scaler.inverse_transform(center.reshape(1, -1)).ravel()
                            else:
                                original_center = center
                            
                            fig.add_trace(go.Scatter(
                                x=[original_center[0]],
                                y=[original_center[1]],
                                mode='markers',
                                marker=dict(
                                    symbol='x',
                                    size=15,
                                    color='black',
                                    line=dict(width=2)
                                ),
                                name=f'Centroid {i}',
                                showlegend=True
                            ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:  # Just one feature
                        fig = px.histogram(
                            cluster_data,
                            x=numeric_cols[0],
                            color='Cluster',
                            title=f'Distribution of {numeric_cols[0]} by Cluster',
                            category_orders={'Cluster': sorted(cluster_data['Cluster'].unique())},
                            color_discrete_sequence=px.colors.qualitative.G10,
                            barmode='overlay',
                            histnorm='percent'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # 3. Cluster profiles (feature means by cluster)
                st.write("#### Cluster Profiles")
                
                cluster_profiles = cluster_data.groupby('Cluster')[numeric_cols].mean()
                
                # Normalize profiles for better visualization
                if standardize == "Yes, standardize (recommended)":
                    # Profiles are already using standardized values
                    profiles_plot = cluster_profiles.copy()
                else:
                    # Standardize just for comparison
                    profiles_plot = (cluster_profiles - data[numeric_cols].mean()) / data[numeric_cols].std()
                
                # Melt for Plotly
                profiles_melted = profiles_plot.reset_index().melt(
                    id_vars=['Cluster'],
                    value_vars=numeric_cols,
                    var_name='Feature',
                    value_name='Normalized Value'
                )
                
                fig = px.line(
                    profiles_melted,
                    x='Feature',
                    y='Normalized Value',
                    color='Cluster',
                    title='Feature Profiles by Cluster (Normalized)',
                    markers=True,
                    category_orders={'Cluster': sorted(profiles_melted['Cluster'].unique())},
                    color_discrete_sequence=px.colors.qualitative.G10
                )
                
                fig.update_layout(
                    xaxis_title="Feature",
                    yaxis_title="Standardized Value",
                    legend_title="Cluster"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 4. Show the raw cluster profiles
                st.write("#### Cluster Centroids (Original Scale)")
                
                # Transform centroids back to original scale if standardized
                if standardize == "Yes, standardize (recommended)":
                    original_centroids = scaler.inverse_transform(kmeans.cluster_centers_)
                    centroids_df = pd.DataFrame(original_centroids, columns=numeric_cols)
                else:
                    centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)
                
                centroids_df.index.name = 'Cluster'
                centroids_df.index = centroids_df.index.astype(int)
                
                st.dataframe(centroids_df.style.format('{:.4f}'), use_container_width=True)
                
                # 5. Cluster interpretations
                st.write("#### Cluster Interpretations")
                
                # Analyze each cluster
                for i in range(k_value):
                    with st.expander(f"Cluster {i} ({cluster_counts[cluster_counts['Cluster'] == i]['Count'].values[0]} points)", expanded=False):
                        # Get the centroid
                        centroid = centroids_df.loc[i]
                        
                        # Compare with overall average
                        overall_avg = data[numeric_cols].mean()
                        
                        # Percent difference
                        pct_diff = ((centroid - overall_avg) / overall_avg) * 100
                        
                        # Find the top differentiating features
                        top_features = pct_diff.abs().sort_values(ascending=False)
                        
                        # Show information
                        st.write("**Top Differentiating Features:**")
                        
                        for feature in top_features.index[:min(5, len(top_features))]:
                            direction = "higher" if centroid[feature] > overall_avg[feature] else "lower"
                            st.write(f"- **{feature}**: {centroid[feature]:.4f} ({abs(pct_diff[feature]):.1f}% {direction} than average)")
                        
                        # Show sample points in this cluster
                        sample_size = min(5, len(cluster_data[cluster_data['Cluster'] == i]))
                        if sample_size > 0:
                            st.write("**Sample Data Points:**")
                            st.dataframe(cluster_data[cluster_data['Cluster'] == i].sample(sample_size))

def hierarchical_clustering(data, numeric_cols, standardize, viz_container):
    # Hierarchical clustering parameters
    st.write("Hierarchical Clustering Parameters:")
    
    n_clusters = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=min(15, len(data) // 20),  # Limit based on data size
        value=3,
        key="hc_clusters"
    )
    
    linkage_method = st.selectbox(
        "Linkage Method",
        options=["ward", "complete", "average", "single"],
        index=0,
        key="hc_linkage"
    )
    
    distance_metric = "euclidean"  # Default for simplicity, could be made selectable
    
    # Run Hierarchical clustering button
    if st.button("Run Hierarchical Clustering", key="run_hc"):
        with st.spinner("Running Hierarchical clustering and visualizing results..."):
            # Extract features
            X = data[numeric_cols].values
            
            # Standardize if selected
            if standardize == "Yes, standardize (recommended)":
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Compute the linkage matrix
            Z = linkage(X_scaled, method=linkage_method, metric=distance_metric)
            
            # Get clusters
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            clusters = model.fit_predict(X_scaled)
            
            # Add clusters to dataframe
            cluster_data = data.copy()
            cluster_data['Cluster'] = clusters
            
            # Save clusters to session state
            if 'clustering_results' not in st.session_state:
                st.session_state.clustering_results = {}
            
            st.session_state.clustering_results['hierarchical'] = {
                'data': cluster_data,
                'method': 'Hierarchical',
                'params': {'n_clusters': n_clusters, 'linkage': linkage_method},
                'columns': numeric_cols,
                'clusters': clusters,
                'model': model,
                'linkage_matrix': Z,
                'scaler': scaler if standardize == "Yes, standardize (recommended)" else None
            }
            
            # Draw visualizations
            with viz_container:
                st.subheader("Hierarchical Clustering Results")
                
                # 1. Show dendrogram
                st.write("#### Dendrogram")
                
                plt.figure(figsize=(12, 8))
                plt.title(f"Hierarchical Clustering Dendrogram (Truncated)", fontsize=15)
                plt.xlabel("Sample index or cluster size", fontsize=12)
                plt.ylabel(f"Distance ({distance_metric})", fontsize=12)
                
                # Plot a truncated dendrogram
                if len(X) > 100:
                    st.info("Dendrogram is truncated due to dataset size.")
                    plt.figure(figsize=(12, 8))
                    dendrogram(
                        Z,
                        truncate_mode='lastp',  # Show only the last p clusters
                        p=30,  # Show last 30 merges
                        show_leaf_counts=True,
                        leaf_font_size=12.,
                        color_threshold=0.7*max(Z[:,2])
                    )
                else:
                    plt.figure(figsize=(12, 8))
                    dendrogram(
                        Z,
                        leaf_rotation=90.,
                        leaf_font_size=8.,
                        color_threshold=0.7*max(Z[:,2])
                    )
                
                # Convert matplotlib figure to Plotly
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=100)
                plt.close()
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode()
                
                st.markdown(f'<img src="data:image/png;base64,{img_str}" width="100%">', unsafe_allow_html=True)
                
                # 2. Show number of points in each cluster
                st.write("#### Cluster Sizes")
                cluster_counts = pd.DataFrame(cluster_data['Cluster'].value_counts()).reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                cluster_counts = cluster_counts.sort_values('Cluster')
                
                fig = px.bar(
                    cluster_counts,
                    x='Cluster',
                    y='Count',
                    color='Cluster',
                    title='Number of Data Points per Cluster'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 3. Perform dimensionality reduction for visualization
                if len(numeric_cols) > 2:
                    st.write("#### 2D Cluster Visualization")
                    st.info("Using PCA to visualize high-dimensional clusters in 2D space.")
                    
                    # PCA for dimensionality reduction
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    # Create a DataFrame for plotting
                    pca_df = pd.DataFrame({
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1],
                        'Cluster': clusters
                    })
                    
                    # Plot clusters
                    fig = px.scatter(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        color='Cluster',
                        title='PCA Projection of Clusters',
                        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                        category_orders={'Cluster': sorted(pca_df['Cluster'].unique())},
                        color_discrete_sequence=px.colors.qualitative.G10
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Variance explained
                    var_explained = pca.explained_variance_ratio_
                    st.write(f"Variance explained: PC1 ({var_explained[0]:.2%}), PC2 ({var_explained[1]:.2%}), "
                             f"Total ({sum(var_explained):.2%})")
                    
                    # Show warning if low variance explained
                    if sum(var_explained) < 0.5:
                        st.warning("⚠️ Low variance explained by the first 2 principal components. "
                                   "This 2D visualization may not accurately represent the actual clusters in higher dimensions.")
                else:
                    # Direct scatter plot for 2 features
                    st.write("#### Cluster Visualization")
                    
                    if len(numeric_cols) == 2:
                        # Just plot the two columns
                        fig = px.scatter(
                            cluster_data,
                            x=numeric_cols[0],
                            y=numeric_cols[1],
                            color='Cluster',
                            title=f'Clusters by {numeric_cols[0]} and {numeric_cols[1]}',
                            category_orders={'Cluster': sorted(cluster_data['Cluster'].unique())},
                            color_discrete_sequence=px.colors.qualitative.G10
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:  # Just one feature
                        fig = px.histogram(
                            cluster_data,
                            x=numeric_cols[0],
                            color='Cluster',
                            title=f'Distribution of {numeric_cols[0]} by Cluster',
                            category_orders={'Cluster': sorted(cluster_data['Cluster'].unique())},
                            color_discrete_sequence=px.colors.qualitative.G10,
                            barmode='overlay',
                            histnorm='percent'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # 4. Cluster profiles (feature means by cluster)
                st.write("#### Cluster Profiles")
                
                cluster_profiles = cluster_data.groupby('Cluster')[numeric_cols].mean()
                
                # Normalize profiles for better visualization
                if standardize == "Yes, standardize (recommended)":
                    # Profiles are already using standardized values
                    profiles_plot = cluster_profiles.copy()
                else:
                    # Standardize just for comparison
                    profiles_plot = (cluster_profiles - data[numeric_cols].mean()) / data[numeric_cols].std()
                
                # Melt for Plotly
                profiles_melted = profiles_plot.reset_index().melt(
                    id_vars=['Cluster'],
                    value_vars=numeric_cols,
                    var_name='Feature',
                    value_name='Normalized Value'
                )
                
                fig = px.line(
                    profiles_melted,
                    x='Feature',
                    y='Normalized Value',
                    color='Cluster',
                    title='Feature Profiles by Cluster (Normalized)',
                    markers=True,
                    category_orders={'Cluster': sorted(profiles_melted['Cluster'].unique())},
                    color_discrete_sequence=px.colors.qualitative.G10
                )
                
                fig.update_layout(
                    xaxis_title="Feature",
                    yaxis_title="Standardized Value",
                    legend_title="Cluster"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 5. Show the raw cluster profiles
                st.write("#### Cluster Centroids (Original Scale)")
                
                # Show centroids
                centroids_df = cluster_profiles.copy()
                
                st.dataframe(centroids_df.style.format('{:.4f}'), use_container_width=True)
                
                # 6. Cluster interpretations
                st.write("#### Cluster Interpretations")
                
                # Analyze each cluster
                for i in range(n_clusters):
                    with st.expander(f"Cluster {i} ({cluster_counts[cluster_counts['Cluster'] == i]['Count'].values[0]} points)", expanded=False):
                        # Get the centroid
                        centroid = centroids_df.loc[i]
                        
                        # Compare with overall average
                        overall_avg = data[numeric_cols].mean()
                        
                        # Percent difference
                        pct_diff = ((centroid - overall_avg) / overall_avg) * 100
                        
                        # Find the top differentiating features
                        top_features = pct_diff.abs().sort_values(ascending=False)
                        
                        # Show information
                        st.write("**Top Differentiating Features:**")
                        
                        for feature in top_features.index[:min(5, len(top_features))]:
                            direction = "higher" if centroid[feature] > overall_avg[feature] else "lower"
                            st.write(f"- **{feature}**: {centroid[feature]:.4f} ({abs(pct_diff[feature]):.1f}% {direction} than average)")
                        
                        # Show sample points in this cluster
                        sample_size = min(5, len(cluster_data[cluster_data['Cluster'] == i]))
                        if sample_size > 0:
                            st.write("**Sample Data Points:**")
                            st.dataframe(cluster_data[cluster_data['Cluster'] == i].sample(sample_size))

def dbscan_clustering(data, numeric_cols, standardize, viz_container):
    # DBSCAN parameters
    st.write("DBSCAN Parameters:")
    
    # Determine a reasonable default for eps
    if standardize == "Yes, standardize (recommended)":
        eps_default = 0.5  # Default for standardized data
    else:
        # Calculate average range to set a reasonable default
        avg_range = np.mean([data[col].max() - data[col].min() for col in numeric_cols])
        eps_default = avg_range * 0.1  # 10% of average range
    
    eps_value = st.slider(
        "Epsilon (neighborhood size)",
        min_value=0.01,
        max_value=float(min(eps_default * 10, 100.0)),  # Limit the max value
        value=float(eps_default),
        step=0.01,
        key="dbscan_eps"
    )
    
    min_samples = st.slider(
        "Minimum Samples",
        min_value=2,
        max_value=50,
        value=5,
        step=1,
        key="dbscan_min_samples"
    )
    
    # Run DBSCAN clustering button
    if st.button("Run DBSCAN Clustering", key="run_dbscan"):
        with st.spinner("Running DBSCAN clustering and visualizing results..."):
            # Extract features
            X = data[numeric_cols].values
            
            # Standardize if selected
            if standardize == "Yes, standardize (recommended)":
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Fit DBSCAN
            dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
            clusters = dbscan.fit_predict(X_scaled)
            
            # Add clusters to dataframe
            cluster_data = data.copy()
            cluster_data['Cluster'] = clusters
            
            # Save clusters to session state
            if 'clustering_results' not in st.session_state:
                st.session_state.clustering_results = {}
            
            st.session_state.clustering_results['dbscan'] = {
                'data': cluster_data,
                'method': 'DBSCAN',
                'params': {'eps': eps_value, 'min_samples': min_samples},
                'columns': numeric_cols,
                'clusters': clusters,
                'model': dbscan,
                'scaler': scaler if standardize == "Yes, standardize (recommended)" else None
            }
            
            # Draw visualizations
            with viz_container:
                st.subheader("DBSCAN Clustering Results")
                
                # Count number of clusters (excluding noise points with cluster -1)
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                n_noise = list(clusters).count(-1)
                
                st.write(f"#### Found {n_clusters} clusters and {n_noise} noise points ({n_noise/len(clusters):.1%} of data)")
                
                # 1. Show number of points in each cluster
                st.write("#### Cluster Sizes")
                cluster_counts = pd.DataFrame(cluster_data['Cluster'].value_counts()).reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                cluster_counts = cluster_counts.sort_values('Cluster')
                
                # Create a custom color map with red for noise points (-1)
                colors = {}
                for cluster in sorted(cluster_data['Cluster'].unique()):
                    if cluster == -1:
                        colors[cluster] = 'red'  # Noise points
                    else:
                        colors[cluster] = px.colors.qualitative.G10[cluster % len(px.colors.qualitative.G10)]
                
                fig = px.bar(
                    cluster_counts,
                    x='Cluster',
                    y='Count',
                    color='Cluster',
                    title='Number of Data Points per Cluster',
                    color_discrete_map=colors
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # If no clusters found, show warning
                if n_clusters == 0:
                    st.error("⚠️ No clusters found! Try adjusting the epsilon or min_samples parameters.")
                    return
                
                # 2. Perform dimensionality reduction for visualization
                if len(numeric_cols) > 2:
                    st.write("#### 2D Cluster Visualization")
                    st.info("Using PCA to visualize high-dimensional clusters in 2D space.")
                    
                    # PCA for dimensionality reduction
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    # Create a DataFrame for plotting
                    pca_df = pd.DataFrame({
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1],
                        'Cluster': clusters
                    })
                    
                    # Plot clusters
                    fig = px.scatter(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        color='Cluster',
                        title='PCA Projection of Clusters',
                        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                        color_discrete_map=colors
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Variance explained
                    var_explained = pca.explained_variance_ratio_
                    st.write(f"Variance explained: PC1 ({var_explained[0]:.2%}), PC2 ({var_explained[1]:.2%}), "
                             f"Total ({sum(var_explained):.2%})")
                    
                    # Show warning if low variance explained
                    if sum(var_explained) < 0.5:
                        st.warning("⚠️ Low variance explained by the first 2 principal components. "
                                   "This 2D visualization may not accurately represent the actual clusters in higher dimensions.")
                else:
                    # Direct scatter plot for 2 features
                    st.write("#### Cluster Visualization")
                    
                    if len(numeric_cols) == 2:
                        # Just plot the two columns
                        fig = px.scatter(
                            cluster_data,
                            x=numeric_cols[0],
                            y=numeric_cols[1],
                            color='Cluster',
                            title=f'Clusters by {numeric_cols[0]} and {numeric_cols[1]}',
                            color_discrete_map=colors
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:  # Just one feature
                        fig = px.histogram(
                            cluster_data,
                            x=numeric_cols[0],
                            color='Cluster',
                            title=f'Distribution of {numeric_cols[0]} by Cluster',
                            color_discrete_map=colors,
                            barmode='overlay',
                            histnorm='percent'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # 3. Cluster profiles (feature means by cluster)
                st.write("#### Cluster Profiles")
                
                # Filter out noise points for profile analysis
                clustered_data = cluster_data[cluster_data['Cluster'] != -1].copy()
                
                if not clustered_data.empty:
                    cluster_profiles = clustered_data.groupby('Cluster')[numeric_cols].mean()
                    
                    # Normalize profiles for better visualization
                    if standardize == "Yes, standardize (recommended)":
                        # Profiles are already using standardized values
                        profiles_plot = cluster_profiles.copy()
                    else:
                        # Standardize just for comparison
                        profiles_plot = (cluster_profiles - data[numeric_cols].mean()) / data[numeric_cols].std()
                    
                    # Melt for Plotly
                    profiles_melted = profiles_plot.reset_index().melt(
                        id_vars=['Cluster'],
                        value_vars=numeric_cols,
                        var_name='Feature',
                        value_name='Normalized Value'
                    )
                    
                    # Custom color map for the profiles
                    cluster_colors = {k: v for k, v in colors.items() if k != -1}
                    
                    fig = px.line(
                        profiles_melted,
                        x='Feature',
                        y='Normalized Value',
                        color='Cluster',
                        title='Feature Profiles by Cluster (Normalized)',
                        markers=True,
                        color_discrete_map=cluster_colors
                    )
                    
                    fig.update_layout(
                        xaxis_title="Feature",
                        yaxis_title="Standardized Value",
                        legend_title="Cluster"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 4. Show the raw cluster profiles
                    st.write("#### Cluster Centroids (Original Scale)")
                    
                    # Show centroids
                    centroids_df = cluster_profiles.copy()
                    
                    st.dataframe(centroids_df.style.format('{:.4f}'), use_container_width=True)
                    
                    # 5. Cluster interpretations
                    st.write("#### Cluster Interpretations")
                    
                    # Analyze each cluster (excluding noise)
                    for i in sorted(cluster_data['Cluster'].unique()):
                        if i != -1:  # Skip noise points
                            cluster_size = cluster_counts[cluster_counts['Cluster'] == i]['Count'].values[0]
                            with st.expander(f"Cluster {i} ({cluster_size} points)", expanded=False):
                                # Get the centroid
                                centroid = centroids_df.loc[i]
                                
                                # Compare with overall average
                                overall_avg = data[numeric_cols].mean()
                                
                                # Percent difference
                                pct_diff = ((centroid - overall_avg) / overall_avg) * 100
                                
                                # Find the top differentiating features
                                top_features = pct_diff.abs().sort_values(ascending=False)
                                
                                # Show information
                                st.write("**Top Differentiating Features:**")
                                
                                for feature in top_features.index[:min(5, len(top_features))]:
                                    direction = "higher" if centroid[feature] > overall_avg[feature] else "lower"
                                    st.write(f"- **{feature}**: {centroid[feature]:.4f} ({abs(pct_diff[feature]):.1f}% {direction} than average)")
                                
                                # Show sample points in this cluster
                                sample_size = min(5, len(cluster_data[cluster_data['Cluster'] == i]))
                                if sample_size > 0:
                                    st.write("**Sample Data Points:**")
                                    st.dataframe(cluster_data[cluster_data['Cluster'] == i].sample(sample_size))
                    
                    # 6. Analyze noise points separately
                    if n_noise > 0:
                        with st.expander(f"Noise Points ({n_noise} points)", expanded=False):
                            noise_data = cluster_data[cluster_data['Cluster'] == -1]
                            
                            # Show statistics of noise points
                            st.write("**Statistics of Noise Points:**")
                            noise_stats = noise_data[numeric_cols].describe().T[['mean', 'min', 'max', 'std']]
                            st.dataframe(noise_stats.style.format('{:.4f}'), use_container_width=True)
                            
                            # Compare with overall data
                            st.write("**Comparison with Overall Data:**")
                            
                            overall_stats = data[numeric_cols].describe().T[['mean', 'min', 'max', 'std']]
                            comparison = pd.DataFrame({
                                'Noise Mean': noise_stats['mean'],
                                'Overall Mean': overall_stats['mean'],
                                'Difference (%)': ((noise_stats['mean'] - overall_stats['mean']) / overall_stats['mean'] * 100)
                            })
                            
                            st.dataframe(comparison.style.format({
                                'Noise Mean': '{:.4f}',
                                'Overall Mean': '{:.4f}',
                                'Difference (%)': '{:.2f}%'
                            }), use_container_width=True)
                            
                            # Show sample noise points
                            sample_size = min(5, len(noise_data))
                            if sample_size > 0:
                                st.write("**Sample Noise Points:**")
                                st.dataframe(noise_data.sample(sample_size))
                else:
                    st.warning("All points were classified as noise. Try reducing epsilon or min_samples.")

def find_elbow_point(x, y):
    """Find the elbow point in a curve using the maximum curvature method."""
    # Need at least 3 points to calculate an elbow
    if len(x) < 3 or len(y) < 3:
        return 1
    
    # Simple approach: find the point with maximum "curvature"
    # by looking at the angle between consecutive line segments
    max_angle = 0
    elbow_idx = 1
    
    # Normalize data to [0,1] range to avoid scale issues
    x_norm = np.array(x)
    y_norm = np.array(y)
    x_range = max(x_norm) - min(x_norm)
    y_range = max(y_norm) - min(y_norm)
    if x_range > 0:
        x_norm = (x_norm - min(x_norm)) / x_range
    if y_range > 0:
        y_norm = (y_norm - min(y_norm)) / y_range
    
    # For each interior point, calculate the angle
    for i in range(1, len(x_norm) - 1):
        # Vectors from current point to neighbors
        v1 = np.array([x_norm[i] - x_norm[i-1], y_norm[i] - y_norm[i-1]])
        v2 = np.array([x_norm[i+1] - x_norm[i], y_norm[i+1] - y_norm[i]])
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0 and v2_norm > 0:
            v1 = v1 / v1_norm
            v2 = v2 / v2_norm
            
            # Calculate the angle (dot product, clamped to avoid numerical issues)
            cosine = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(cosine)
            
            # Find the maximum angle change
            if angle > max_angle:
                max_angle = angle
                elbow_idx = i
    
    return x[elbow_idx]


    # What's next section
    st.markdown("---")
    st.markdown("## What's Next?")
    st.info("👉 Proceed to the **AI Insights** page to discover intelligent patterns and anomalies in your data.")

if __name__ == "__main__":
    main()