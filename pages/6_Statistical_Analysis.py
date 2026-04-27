import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy import stats
import io
import json
import sys
import os

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
    st.title("📊 Statistical Analysis")
    
    # Check if data exists in session state
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ Please upload a Dataset in the **Data Upload** page.")
        st.stop()
    
    # Get data from session state
    data = st.session_state.data

    # Sidebar for navigation within this page
    analysis_type = st.sidebar.radio(
        "Choose Analysis Type",
        ["Descriptive Statistics", "Hypothesis Testing", "Regression Analysis", "Time Series Analysis"]
    )
    
    # Main area
    if analysis_type == "Descriptive Statistics":
        descriptive_statistics(data)
    elif analysis_type == "Hypothesis Testing":
        hypothesis_testing(data)
    elif analysis_type == "Regression Analysis":
        regression_analysis(data)
    elif analysis_type == "Time Series Analysis":
        time_series_analysis(data)

def descriptive_statistics(data):
    st.header("📋 Descriptive Statistics")
    
    # Select columns for analysis
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns available for statistical analysis.")
        return
        
    # Create tabs for different aspects of descriptive statistics
    tab1, tab2, tab3 = st.tabs(["Summary Statistics", "Distribution Analysis", "Box Plots"])
    
    with tab1:
        st.subheader("Summary Statistics")
        
        # Allow user to select columns
        selected_cols = st.multiselect(
            "Select columns for summary statistics:",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))]
        )
        
        if selected_cols:
            # Display summary statistics
            summary_stats = data[selected_cols].describe().T
            
            # Add additional statistics
            summary_stats['median'] = data[selected_cols].median()
            summary_stats['mode'] = data[selected_cols].mode().iloc[0]
            summary_stats['variance'] = data[selected_cols].var()
            summary_stats['skewness'] = data[selected_cols].skew()
            summary_stats['kurtosis'] = data[selected_cols].kurtosis()
            
            # Format for display
            summary_stats = summary_stats.round(2)
            st.dataframe(summary_stats)
            
            # Option to download
            st.download_button(
                "Download Summary Statistics",
                data=summary_stats.to_csv().encode('utf-8'),
                file_name="summary_statistics.csv",
                mime="text/csv"
            )
    
    with tab2:
        st.subheader("Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select column for histogram
            hist_col = st.selectbox(
                "Select column for histogram:",
                numeric_cols
            )
            
            if hist_col:
                # Create histogram with density curve
                fig = px.histogram(
                    data, 
                    x=hist_col,
                    marginal="box",
                    title=f"Distribution of {hist_col}",
                    opacity=0.7,
                    histnorm="probability density"
                )
                
                fig.update_layout(
                    xaxis_title=hist_col,
                    yaxis_title="Frequency",
                    showlegend=False
                )
                
                st.plotly_chart(fig)
                
                # Normality test
                stat, p_value = stats.shapiro(data[hist_col].dropna())
                
                # Display normality test results
                st.write("**Normality Test (Shapiro-Wilk)**")
                st.write(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
                
                if p_value < 0.05:
                    st.write("The distribution is **not normal** (p < 0.05)")
                else:
                    st.write("The distribution appears to be **normal** (p >= 0.05)")
        
        with col2:
            # Q-Q Plot for normality check
            qq_col = st.selectbox(
                "Select column for Q-Q plot:",
                numeric_cols,
                key="qq_col"
            )
            
            if qq_col:
                # Create Q-Q plot
                fig = plt.figure(figsize=(10, 6))
                stats.probplot(data[qq_col].dropna(), plot=plt)
                plt.title(f"Q-Q Plot for {qq_col}")
                st.pyplot(fig)
                
                # Display basic statistics
                st.write("**Basic Statistics**")
                st.write(f"Mean: {data[qq_col].mean():.2f}")
                st.write(f"Median: {data[qq_col].median():.2f}")
                st.write(f"Skewness: {data[qq_col].skew():.2f}")
                st.write(f"Kurtosis: {data[qq_col].kurtosis():.2f}")
    
    with tab3:
        st.subheader("Box Plots")
        
        # Select columns for box plot
        box_cols = st.multiselect(
            "Select columns for box plot:",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))],
            key="box_cols"
        )
        
        if box_cols:
            # Create box plot
            fig = px.box(
                data_frame=data,
                y=box_cols,
                title="Box Plots for Selected Variables",
                points="all"
            )
            
            st.plotly_chart(fig)
            
            # Optional: Add categorical variable to compare
            if categorical_cols:
                st.write("**Compare by Category**")
                cat_col = st.selectbox(
                    "Select categorical variable for comparison:",
                    categorical_cols
                )
                
                num_col = st.selectbox(
                    "Select numeric variable to compare:",
                    numeric_cols
                )
                
                # Check if too many categories
                if data[cat_col].nunique() > 15:
                    st.warning(f"Too many categories in {cat_col} ({data[cat_col].nunique()}). Consider using top categories.")
                    top_n = st.slider("Number of top categories to display:", 2, 15, 5)
                    top_cats = data[cat_col].value_counts().nlargest(top_n).index.tolist()
                    filtered_data = data[data[cat_col].isin(top_cats)]
                else:
                    filtered_data = data
                
                # Create box plot by category
                fig = px.box(
                    data_frame=filtered_data,
                    x=cat_col,
                    y=num_col,
                    title=f"Box Plot of {num_col} by {cat_col}",
                    color=cat_col
                )
                
                st.plotly_chart(fig)

def hypothesis_testing(data):
    st.header("🧪 Hypothesis Testing")
    
    # Get numeric and categorical columns
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns available for hypothesis testing.")
        return
    
    # Create tabs for different types of tests
    tab1, tab2, tab3, tab4 = st.tabs([
        "t-Tests", 
        "ANOVA", 
        "Chi-Square Test", 
        "Correlation Tests"
    ])
    
    with tab1:
        st.subheader("t-Tests")
        
        # Options for different t-tests
        t_test_type = st.radio(
            "Select t-test type:",
            ["One-Sample t-Test", "Two-Sample t-Test", "Paired t-Test"]
        )
        
        if t_test_type == "One-Sample t-Test":
            # One-sample t-test
            col = st.selectbox("Select column:", numeric_cols, key="one_sample_col")
            
            # Hypothesized mean
            mu = st.number_input(
                "Hypothesized mean (μ):",
                value=round(float(data[col].mean()), 2)
            )
            
            # Run test when button is clicked
            if st.button("Run One-Sample t-Test"):
                # Perform t-test
                result = stats.ttest_1samp(data[col].dropna(), mu)
                
                # Display results
                st.write("### One-Sample t-Test Results")
                st.write(f"**Null Hypothesis (H₀):** The mean of {col} is equal to {mu}")
                st.write(f"**Alternative Hypothesis (H₁):** The mean of {col} is not equal to {mu}")
                
                st.write(f"**t-statistic:** {result.statistic:.4f}")
                st.write(f"**p-value:** {result.pvalue:.4f}")
                
                # Interpretation
                alpha = 0.05
                if result.pvalue < alpha:
                    st.write(f"**Conclusion:** Reject the null hypothesis (p < {alpha})")
                    st.write(f"There is significant evidence that the mean of {col} is not equal to {mu}.")
                else:
                    st.write(f"**Conclusion:** Fail to reject the null hypothesis (p >= {alpha})")
                    st.write(f"There is insufficient evidence that the mean of {col} is different from {mu}.")
                
                # Visualize
                fig = plt.figure(figsize=(10, 6))
                plt.hist(data[col].dropna(), bins=30, alpha=0.7, color='skyblue')
                plt.axvline(mu, color='red', linestyle='dashed', linewidth=2, label=f'Hypothesized mean: {mu}')
                plt.axvline(data[col].mean(), color='green', linestyle='dashed', linewidth=2, label=f'Sample mean: {data[col].mean():.2f}')
                plt.legend()
                plt.title(f'Distribution of {col} with Hypothesized Mean')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                st.pyplot(fig)
        
        elif t_test_type == "Two-Sample t-Test":
            st.write("Compare means between two groups")
            
            # Select numeric column
            num_col = st.selectbox("Select numeric column:", numeric_cols, key="two_sample_num")
            
            # Select categorical column for grouping
            cat_col = st.selectbox(
                "Select categorical column for grouping:",
                categorical_cols,
                key="two_sample_cat"
            )
            
            if data[cat_col].nunique() < 2:
                st.error(f"The column '{cat_col}' needs at least 2 distinct values for a two-sample test.")
            elif data[cat_col].nunique() > 10:
                st.warning(f"The column '{cat_col}' has {data[cat_col].nunique()} distinct values. Select two groups for comparison.")
                
                # Get the top categories
                top_cats = data[cat_col].value_counts().nlargest(10).index.tolist()
                
                # Let user select two groups
                group1 = st.selectbox("Select first group:", top_cats, index=0)
                group2 = st.selectbox("Select second group:", top_cats, index=min(1, len(top_cats)-1))
                
                # Run test when button is clicked
                if st.button("Run Two-Sample t-Test"):
                    # Get data for each group
                    data1 = data[data[cat_col] == group1][num_col].dropna()
                    data2 = data[data[cat_col] == group2][num_col].dropna()
                    
                    # Check if enough data
                    if len(data1) < 2 or len(data2) < 2:
                        st.error("Not enough data in one or both groups after removing missing values.")
                    else:
                        # Perform variance test to decide equal_var parameter
                        var_test = stats.levene(data1, data2)
                        equal_var = var_test.pvalue >= 0.05
                        
                        # Perform t-test
                        result = stats.ttest_ind(data1, data2, equal_var=equal_var)
                        
                        # Display results
                        st.write("### Two-Sample t-Test Results")
                        st.write(f"**Null Hypothesis (H₀):** The means of {num_col} are equal between {group1} and {group2}")
                        st.write(f"**Alternative Hypothesis (H₁):** The means of {num_col} are different between {group1} and {group2}")
                        
                        st.write(f"**t-statistic:** {result.statistic:.4f}")
                        st.write(f"**p-value:** {result.pvalue:.4f}")
                        
                        if not equal_var:
                            st.write("*Note: Unequal variances detected, Welch's t-test was used.*")
                        
                        # Interpretation
                        alpha = 0.05
                        if result.pvalue < alpha:
                            st.write(f"**Conclusion:** Reject the null hypothesis (p < {alpha})")
                            st.write(f"There is significant evidence that the means of {num_col} differ between {group1} and {group2}.")
                        else:
                            st.write(f"**Conclusion:** Fail to reject the null hypothesis (p >= {alpha})")
                            st.write(f"There is insufficient evidence that the means of {num_col} differ between {group1} and {group2}.")
                        
                        # Visualize
                        fig = plt.figure(figsize=(10, 6))
                        plt.boxplot([data1, data2], labels=[group1, group2])
                        plt.title(f'Comparison of {num_col} between {group1} and {group2}')
                        plt.ylabel(num_col)
                        st.pyplot(fig)
                        
                        # Show descriptive statistics
                        st.write("### Group Statistics")
                        stats_df = pd.DataFrame({
                            'Group': [group1, group2],
                            'Count': [len(data1), len(data2)],
                            'Mean': [data1.mean(), data2.mean()],
                            'Std Dev': [data1.std(), data2.std()],
                            'Min': [data1.min(), data2.min()],
                            'Max': [data1.max(), data2.max()]
                        })
                        
                        st.dataframe(stats_df.round(2))
            else:
                # If there are just a few categories, let user select specific ones
                groups = data[cat_col].unique().tolist()
                
                group1 = st.selectbox("Select first group:", groups, index=0)
                group2 = st.selectbox("Select second group:", groups, index=min(1, len(groups)-1))
                
                # Run test when button is clicked
                if st.button("Run Two-Sample t-Test"):
                    # Get data for each group
                    data1 = data[data[cat_col] == group1][num_col].dropna()
                    data2 = data[data[cat_col] == group2][num_col].dropna()
                    
                    # Check if enough data
                    if len(data1) < 2 or len(data2) < 2:
                        st.error("Not enough data in one or both groups after removing missing values.")
                    else:
                        # Perform variance test to decide equal_var parameter
                        var_test = stats.levene(data1, data2)
                        equal_var = var_test.pvalue >= 0.05
                        
                        # Perform t-test
                        result = stats.ttest_ind(data1, data2, equal_var=equal_var)
                        
                        # Display results
                        st.write("### Two-Sample t-Test Results")
                        st.write(f"**Null Hypothesis (H₀):** The means of {num_col} are equal between {group1} and {group2}")
                        st.write(f"**Alternative Hypothesis (H₁):** The means of {num_col} are different between {group1} and {group2}")
                        
                        st.write(f"**t-statistic:** {result.statistic:.4f}")
                        st.write(f"**p-value:** {result.pvalue:.4f}")
                        
                        if not equal_var:
                            st.write("*Note: Unequal variances detected, Welch's t-test was used.*")
                        
                        # Interpretation
                        alpha = 0.05
                        if result.pvalue < alpha:
                            st.write(f"**Conclusion:** Reject the null hypothesis (p < {alpha})")
                            st.write(f"There is significant evidence that the means of {num_col} differ between {group1} and {group2}.")
                        else:
                            st.write(f"**Conclusion:** Fail to reject the null hypothesis (p >= {alpha})")
                            st.write(f"There is insufficient evidence that the means of {num_col} differ between {group1} and {group2}.")
                        
                        # Visualize
                        fig = plt.figure(figsize=(10, 6))
                        plt.boxplot([data1, data2], labels=[group1, group2])
                        plt.title(f'Comparison of {num_col} between {group1} and {group2}')
                        plt.ylabel(num_col)
                        st.pyplot(fig)
                        
                        # Show descriptive statistics
                        st.write("### Group Statistics")
                        stats_df = pd.DataFrame({
                            'Group': [group1, group2],
                            'Count': [len(data1), len(data2)],
                            'Mean': [data1.mean(), data2.mean()],
                            'Std Dev': [data1.std(), data2.std()],
                            'Min': [data1.min(), data2.min()],
                            'Max': [data1.max(), data2.max()]
                        })
                        
                        st.dataframe(stats_df.round(2))
        
        elif t_test_type == "Paired t-Test":
            st.write("Compare paired measurements (before/after, left/right, etc.)")
            
            # Select two numeric columns
            col1 = st.selectbox("Select first column:", numeric_cols, key="paired_col1")
            col2 = st.selectbox("Select second column:", numeric_cols, key="paired_col2", index=min(1, len(numeric_cols)-1))
            
            # Run test when button is clicked
            if st.button("Run Paired t-Test"):
                # Get data with non-missing values in both columns
                paired_data = data[[col1, col2]].dropna()
                
                # Check if enough data
                if len(paired_data) < 2:
                    st.error("Not enough paired data after removing missing values.")
                else:
                    # Perform paired t-test
                    result = stats.ttest_rel(paired_data[col1], paired_data[col2])
                    
                    # Display results
                    st.write("### Paired t-Test Results")
                    st.write(f"**Null Hypothesis (H₀):** The mean difference between {col1} and {col2} is zero")
                    st.write(f"**Alternative Hypothesis (H₁):** The mean difference between {col1} and {col2} is not zero")
                    
                    st.write(f"**t-statistic:** {result.statistic:.4f}")
                    st.write(f"**p-value:** {result.pvalue:.4f}")
                    
                    # Interpretation
                    alpha = 0.05
                    mean_diff = (paired_data[col1] - paired_data[col2]).mean()
                    
                    if result.pvalue < alpha:
                        st.write(f"**Conclusion:** Reject the null hypothesis (p < {alpha})")
                        st.write(f"There is significant evidence that there is a difference between {col1} and {col2}.")
                    else:
                        st.write(f"**Conclusion:** Fail to reject the null hypothesis (p >= {alpha})")
                        st.write(f"There is insufficient evidence that there is a difference between {col1} and {col2}.")
                    
                    # Visualize
                    fig = plt.figure(figsize=(10, 6))
                    plt.scatter(paired_data[col1], paired_data[col2], alpha=0.5)
                    
                    # Add diagonal line
                    min_val = min(paired_data[col1].min(), paired_data[col2].min())
                    max_val = max(paired_data[col1].max(), paired_data[col2].max())
                    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
                    
                    plt.title(f'Scatter Plot of {col1} vs {col2}')
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    plt.axis('equal')
                    st.pyplot(fig)
                    
                    # Show paired differences
                    st.write("### Paired Differences")
                    paired_data['Difference'] = paired_data[col1] - paired_data[col2]
                    
                    fig = plt.figure(figsize=(10, 6))
                    plt.hist(paired_data['Difference'], bins=20, alpha=0.7, color='skyblue')
                    plt.axvline(0, color='red', linestyle='dashed', linewidth=2, label='No difference')
                    plt.axvline(mean_diff, color='green', linestyle='dashed', linewidth=2, label=f'Mean difference: {mean_diff:.2f}')
                    plt.legend()
                    plt.title(f'Distribution of Differences ({col1} - {col2})')
                    plt.xlabel('Difference')
                    plt.ylabel('Frequency')
                    st.pyplot(fig)
    
    with tab2:
        st.subheader("ANOVA")
        
        # ANOVA - Compare means across multiple groups
        st.write("Compare means across multiple groups")
        
        # Select numeric column for values
        num_col = st.selectbox("Select numeric column:", numeric_cols, key="anova_num_col")
        
        # Select categorical column for grouping
        cat_col = st.selectbox(
            "Select categorical column for grouping:",
            categorical_cols,
            key="anova_cat_col"
        )
        
        # Check if column has appropriate number of groups
        if data[cat_col].nunique() < 3:
            st.warning(f"The column '{cat_col}' has fewer than 3 distinct values. ANOVA is typically used for 3 or more groups. Consider using a t-test instead.")
        
        # Run test when button is clicked
        if st.button("Run ANOVA Test"):
            # Get groups
            groups = []
            labels = []
            
            # Limit to top 10 groups if there are many
            if data[cat_col].nunique() > 10:
                st.warning(f"The column '{cat_col}' has {data[cat_col].nunique()} distinct values. Analysis will be limited to the 10 most frequent groups.")
                top_groups = data[cat_col].value_counts().nlargest(10).index.tolist()
                
                for group in top_groups:
                    group_data = data[data[cat_col] == group][num_col].dropna()
                    if len(group_data) > 0:
                        groups.append(group_data)
                        labels.append(str(group))
            else:
                for group in data[cat_col].unique():
                    group_data = data[data[cat_col] == group][num_col].dropna()
                    if len(group_data) > 0:
                        groups.append(group_data)
                        labels.append(str(group))
            
            # Check if enough groups with data
            if len(groups) < 2:
                st.error("Not enough groups with data after removing missing values. ANOVA requires at least 2 groups.")
            else:
                # Perform ANOVA
                result = stats.f_oneway(*groups)
                
                # Display results
                st.write("### ANOVA Results")
                st.write(f"**Null Hypothesis (H₀):** The means of {num_col} are equal across all groups of {cat_col}")
                st.write(f"**Alternative Hypothesis (H₁):** At least one group mean is different")
                
                st.write(f"**F-statistic:** {result.statistic:.4f}")
                st.write(f"**p-value:** {result.pvalue:.4f}")
                
                # Interpretation
                alpha = 0.05
                if result.pvalue < alpha:
                    st.write(f"**Conclusion:** Reject the null hypothesis (p < {alpha})")
                    st.write(f"There is significant evidence that the means of {num_col} differ across at least some groups of {cat_col}.")
                else:
                    st.write(f"**Conclusion:** Fail to reject the null hypothesis (p >= {alpha})")
                    st.write(f"There is insufficient evidence that the means of {num_col} differ across groups of {cat_col}.")
                
                # Visualize
                fig = plt.figure(figsize=(12, 6))
                plt.boxplot(groups, labels=labels)
                plt.title(f'Comparison of {num_col} across {cat_col} Groups')
                plt.ylabel(num_col)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Show descriptive statistics
                st.write("### Group Statistics")
                stats_dict = {
                    'Group': labels,
                    'Count': [len(g) for g in groups],
                    'Mean': [g.mean() for g in groups],
                    'Std Dev': [g.std() for g in groups],
                    'Min': [g.min() for g in groups],
                    'Max': [g.max() for g in groups]
                }
                
                stats_df = pd.DataFrame(stats_dict)
                st.dataframe(stats_df.round(2))
                
                # Post-hoc analysis if ANOVA is significant
                if result.pvalue < alpha and len(groups) > 2:
                    st.write("### Post-hoc Analysis (Tukey's HSD)")
                    st.write("Since the ANOVA test is significant, we can perform post-hoc analysis to identify which specific groups differ.")
                    
                    # Create a DataFrame for post-hoc analysis
                    posthoc_data = data[[cat_col, num_col]].dropna()
                    posthoc_data = posthoc_data[posthoc_data[cat_col].isin(labels)]
                    
                    # Perform Tukey's test
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    tukey = pairwise_tukeyhsd(
                        posthoc_data[num_col], 
                        posthoc_data[cat_col],
                        alpha=0.05
                    )
                    
                    # Display results in a readable format
                    tukey_df = pd.DataFrame(
                        data=tukey._results_table.data[1:],
                        columns=tukey._results_table.data[0]
                    )
                    
                    st.dataframe(tukey_df)
    
    with tab3:
        st.subheader("Chi-Square Test")
        
        # Chi-Square Test - Test for independence between categorical variables
        st.write("Test for independence between categorical variables")
        
        # Select categorical columns
        available_cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(available_cat_cols) < 2:
            st.warning("Need at least two categorical columns for chi-square test.")
        else:
            col1 = st.selectbox("Select first categorical column:", available_cat_cols, key="chi_col1")
            col2 = st.selectbox("Select second categorical column:", available_cat_cols, key="chi_col2", index=min(1, len(available_cat_cols)-1))
            
            # Run test when button is clicked
            if st.button("Run Chi-Square Test"):
                # Create contingency table
                contingency_table = pd.crosstab(data[col1], data[col2])
                
                # Perform chi-square test
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                
                # Display results
                st.write("### Chi-Square Test Results")
                st.write(f"**Null Hypothesis (H₀):** {col1} and {col2} are independent (no association)")
                st.write(f"**Alternative Hypothesis (H₁):** {col1} and {col2} are dependent (associated)")
                
                st.write(f"**Chi-square statistic:** {chi2:.4f}")
                st.write(f"**p-value:** {p:.4f}")
                st.write(f"**Degrees of freedom:** {dof}")
                
                # Interpretation
                alpha = 0.05
                if p < alpha:
                    st.write(f"**Conclusion:** Reject the null hypothesis (p < {alpha})")
                    st.write(f"There is significant evidence of an association between {col1} and {col2}.")
                else:
                    st.write(f"**Conclusion:** Fail to reject the null hypothesis (p >= {alpha})")
                    st.write(f"There is insufficient evidence of an association between {col1} and {col2}.")
                
                # Display contingency table
                st.write("### Contingency Table (Observed Counts)")
                
                # Check if table is too large
                if contingency_table.shape[0] > 10 or contingency_table.shape[1] > 10:
                    st.warning(f"Large contingency table ({contingency_table.shape[0]}×{contingency_table.shape[1]}). Consider grouping categories.")
                
                # Show heat map for better visualization
                fig = px.imshow(
                    contingency_table,
                    labels=dict(x=col2, y=col1, color="Count"),
                    title=f"Heatmap of {col1} vs {col2}",
                    text_auto=True,
                    aspect="auto"
                )
                st.plotly_chart(fig)
                
                # Show numerical contingency table
                st.dataframe(contingency_table)
                
                # Display expected frequencies
                st.write("### Expected Frequencies (if variables were independent)")
                expected_df = pd.DataFrame(
                    expected,
                    index=contingency_table.index,
                    columns=contingency_table.columns
                )
                st.dataframe(expected_df.round(2))
                
                # Optional: Show percentage contribution to chi-square
                st.write("### Contribution to Chi-Square by Cell")
                
                chi2_contribution = (contingency_table - expected)**2 / expected
                chi2_contribution_percentage = 100 * chi2_contribution / chi2
                
                fig = px.imshow(
                    chi2_contribution_percentage,
                    labels=dict(x=col2, y=col1, color="% Contribution"),
                    title=f"Percentage Contribution to Chi-Square Statistic",
                    text_auto='.1f',
                    aspect="auto"
                )
                st.plotly_chart(fig)
    
    with tab4:
        st.subheader("Correlation Tests")
        
        # Correlation Tests - Test for correlations between variables
        st.write("Test for correlations between variables")
        
        # Select correlation type
        corr_type = st.radio(
            "Select correlation test type:",
            ["Pearson (linear)", "Spearman (rank)", "Kendall (rank)"]
        )
        
        # Map selection to method
        method_map = {
            "Pearson (linear)": "pearson",
            "Spearman (rank)": "spearman",
            "Kendall (rank)": "kendall"
        }
        
        method = method_map[corr_type]
        
        # Select variables
        col1 = st.selectbox("Select first variable:", numeric_cols, key="corr_col1")
        col2 = st.selectbox("Select second variable:", numeric_cols, key="corr_col2", index=min(1, len(numeric_cols)-1))
        
        # Run test when button is clicked
        if st.button("Run Correlation Test"):
            # Get data without missing values
            valid_data = data[[col1, col2]].dropna()
            
            # Calculate correlation
            if method == "pearson":
                corr, p_value = stats.pearsonr(valid_data[col1], valid_data[col2])
                test_name = "Pearson's Correlation"
            elif method == "spearman":
                corr, p_value = stats.spearmanr(valid_data[col1], valid_data[col2])
                test_name = "Spearman's Rank Correlation"
            else:  # kendall
                corr, p_value = stats.kendalltau(valid_data[col1], valid_data[col2])
                test_name = "Kendall's Tau Correlation"
            
            # Display results
            st.write(f"### {test_name} Results")
            st.write(f"**Null Hypothesis (H₀):** There is no correlation between {col1} and {col2}")
            st.write(f"**Alternative Hypothesis (H₁):** There is a correlation between {col1} and {col2}")
            
            st.write(f"**Correlation coefficient:** {corr:.4f}")
            st.write(f"**p-value:** {p_value:.4f}")
            
            # Interpretation
            alpha = 0.05
            if p_value < alpha:
                st.write(f"**Conclusion:** Reject the null hypothesis (p < {alpha})")
                
                if corr > 0:
                    strength = "weak" if corr < 0.3 else "moderate" if corr < 0.7 else "strong"
                    st.write(f"There is significant evidence of a **positive {strength}** correlation between {col1} and {col2}.")
                else:
                    strength = "weak" if corr > -0.3 else "moderate" if corr > -0.7 else "strong"
                    st.write(f"There is significant evidence of a **negative {strength}** correlation between {col1} and {col2}.")
            else:
                st.write(f"**Conclusion:** Fail to reject the null hypothesis (p >= {alpha})")
                st.write(f"There is insufficient evidence of a correlation between {col1} and {col2}.")
            
            # Visualize
            fig = plt.figure(figsize=(10, 6))
            plt.scatter(valid_data[col1], valid_data[col2], alpha=0.5)
            
            # Add regression line for Pearson (linear correlation)
            if method == "pearson":
                slope, intercept = np.polyfit(valid_data[col1], valid_data[col2], 1)
                plt.plot(valid_data[col1], slope * valid_data[col1] + intercept, color='red')
                
            plt.title(f'Scatter Plot of {col1} vs {col2}')
            plt.xlabel(col1)
            plt.ylabel(col2)
            st.pyplot(fig)

def regression_analysis(data):
    st.header("📈 Regression Analysis")
    
    # Get numeric and categorical columns
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns for regression analysis.")
        return
    
    # Create tabs for different types of regression
    tab1, tab2, tab3 = st.tabs(["Linear Regression", "Multiple Regression", "Logistic Regression"])
    
    with tab1:
        st.subheader("Linear Regression")
        
        # Select variables
        x_var = st.selectbox("Select independent variable (X):", numeric_cols, key="linear_x")
        y_var = st.selectbox(
            "Select dependent variable (Y):",
            numeric_cols,
            key="linear_y",
            index=min(1, len(numeric_cols)-1)
        )
        
        # Run analysis when button is clicked
        if st.button("Run Linear Regression"):
            # Get data without missing values
            valid_data = data[[x_var, y_var]].dropna()
            
            # Check if enough data
            if len(valid_data) < 2:
                st.error("Not enough data for regression after removing missing values.")
            else:
                # Fit linear regression model
                X = valid_data[x_var].values.reshape(-1, 1)
                y = valid_data[y_var].values
                
                model = sm.OLS(y, sm.add_constant(X)).fit()
                
                # Generate predictions
                valid_data['predicted'] = model.predict(sm.add_constant(X))
                
                # Calculate metrics
                r2 = model.rsquared
                adj_r2 = model.rsquared_adj
                rmse = np.sqrt(np.mean((y - valid_data['predicted'])**2))
                
                # Display results
                st.write("### Linear Regression Results")
                st.write(f"**Model:** {y_var} = {model.params[0]:.4f} + {model.params[1]:.4f} × {x_var}")
                
                st.write(f"**R²:** {r2:.4f}")
                st.write(f"**Adjusted R²:** {adj_r2:.4f}")
                st.write(f"**RMSE:** {rmse:.4f}")
                
                # Interpretation
                if r2 < 0.3:
                    r2_interpretation = "weak"
                elif r2 < 0.7:
                    r2_interpretation = "moderate"
                else:
                    r2_interpretation = "strong"
                
                st.write(f"**Interpretation:** The model explains {r2:.1%} of the variance in {y_var}, "
                         f"indicating a {r2_interpretation} relationship.")
                
                # Visualize
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Scatter plot with regression line
                ax1.scatter(X, y, alpha=0.5)
                ax1.plot(X, valid_data['predicted'], color='red')
                ax1.set_title(f'Linear Regression: {y_var} vs {x_var}')
                ax1.set_xlabel(x_var)
                ax1.set_ylabel(y_var)
                
                # Residual plot
                residuals = y - valid_data['predicted']
                ax2.scatter(valid_data['predicted'], residuals, alpha=0.5)
                ax2.axhline(y=0, color='red', linestyle='--')
                ax2.set_title('Residual Plot')
                ax2.set_xlabel('Predicted Values')
                ax2.set_ylabel('Residuals')
                
                st.pyplot(fig)
                
                # Regression diagnostics
                st.write("### Regression Diagnostics")
                
                # Display model summary
                model_summary = model.summary()
                st.text(str(model_summary))
                
                # Check residuals for normality
                fig = plt.figure(figsize=(10, 6))
                stats.probplot(residuals, plot=plt)
                plt.title('Q-Q Plot of Residuals')
                st.pyplot(fig)
                
                # Hypothesis test for slope
                st.write("### Hypothesis Test for Slope")
                st.write(f"**Null Hypothesis (H₀):** The slope coefficient for {x_var} is zero (no effect)")
                st.write(f"**Alternative Hypothesis (H₁):** The slope coefficient for {x_var} is not zero (there is an effect)")
                
                slope_p_value = model.pvalues[1]
                st.write(f"**p-value:** {slope_p_value:.4f}")
                
                # Interpretation
                alpha = 0.05
                if slope_p_value < alpha:
                    st.write(f"**Conclusion:** Reject the null hypothesis (p < {alpha})")
                    st.write(f"There is significant evidence that {x_var} has an effect on {y_var}.")
                else:
                    st.write(f"**Conclusion:** Fail to reject the null hypothesis (p >= {alpha})")
                    st.write(f"There is insufficient evidence that {x_var} has an effect on {y_var}.")
    
    with tab2:
        st.subheader("Multiple Regression")
        
        # Select dependent variable
        y_var = st.selectbox("Select dependent variable (Y):", numeric_cols, key="multi_y")
        
        # Select independent variables
        remaining_cols = [col for col in numeric_cols if col != y_var]
        x_vars = st.multiselect(
            "Select independent variables (X):",
            remaining_cols,
            default=remaining_cols[:min(3, len(remaining_cols))]
        )
        
        # Include categorical variables
        include_categorical = st.checkbox("Include categorical variables", value=False)
        cat_vars = []
        
        if include_categorical and categorical_cols:
            cat_vars = st.multiselect(
                "Select categorical variables:",
                categorical_cols
            )
        
        # Run analysis when button is clicked
        if st.button("Run Multiple Regression") and x_vars:
            # Create a copy of the data for analysis
            all_vars = [y_var] + x_vars + cat_vars
            reg_data = data[all_vars].copy()
            
            # Handle categorical variables
            formula = f"{y_var} ~ "
            
            # Add numeric variables
            if x_vars:
                formula += " + ".join(x_vars)
            
            # Add categorical variables
            if cat_vars:
                if x_vars:
                    formula += " + "
                formula += " + ".join([f"C({var})" for var in cat_vars])
            
            # Fit the model
            try:
                model = ols(formula, data=reg_data).fit()
                
                # Display results
                st.write("### Multiple Regression Results")
                
                # Model metrics
                r2 = model.rsquared
                adj_r2 = model.rsquared_adj
                
                st.write(f"**R²:** {r2:.4f}")
                st.write(f"**Adjusted R²:** {adj_r2:.4f}")
                
                # Interpretation
                if r2 < 0.3:
                    r2_interpretation = "weak"
                elif r2 < 0.7:
                    r2_interpretation = "moderate"
                else:
                    r2_interpretation = "strong"
                
                st.write(f"**Interpretation:** The model explains {r2:.1%} of the variance in {y_var}, "
                         f"indicating a {r2_interpretation} relationship.")
                
                # Display model summary
                model_summary = model.summary()
                st.text(str(model_summary))
                
                # Variable importance
                st.write("### Variable Importance")
                
                # Get standardized coefficients
                params = pd.DataFrame({'coef': model.params.values[1:]}, index=model.params.index[1:])
                params['abs_coef'] = np.abs(params['coef'])
                params = params.sort_values('abs_coef', ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    params, 
                    y=params.index, 
                    x='coef',
                    orientation='h',
                    title='Coefficient Values (Variable Importance)',
                    color='coef',
                    color_continuous_scale='RdBu_r'
                )
                
                st.plotly_chart(fig)
                
                # Residual analysis
                st.write("### Residual Analysis")
                
                # Add predictions and calculate residuals
                reg_data['predicted'] = model.predict(reg_data)
                reg_data['residuals'] = reg_data[y_var] - reg_data['predicted']
                
                # Create residual plots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Residuals vs Predicted
                ax1.scatter(reg_data['predicted'], reg_data['residuals'], alpha=0.5)
                ax1.axhline(y=0, color='red', linestyle='--')
                ax1.set_title('Residuals vs Predicted Values')
                ax1.set_xlabel('Predicted Values')
                ax1.set_ylabel('Residuals')
                
                # Histogram of residuals
                ax2.hist(reg_data['residuals'], bins=20, alpha=0.7)
                ax2.set_title('Distribution of Residuals')
                ax2.set_xlabel('Residuals')
                ax2.set_ylabel('Frequency')
                
                st.pyplot(fig)
                
                # Check for multicollinearity
                if len(x_vars) > 1:
                    st.write("### Multicollinearity Check")
                    
                    # Calculate correlation matrix
                    corr_matrix = reg_data[x_vars].corr()
                    
                    # Create heatmap
                    fig = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        title='Correlation Matrix of Independent Variables',
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Calculate VIF (Variance Inflation Factor)
                    X = sm.add_constant(reg_data[x_vars])
                    vif_data = pd.DataFrame()
                    vif_data["Variable"] = X.columns
                    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                    
                    # Remove constant
                    vif_data = vif_data[vif_data['Variable'] != 'const']
                    
                    st.write("**Variance Inflation Factors (VIF)**")
                    st.write("VIF > 5 indicates potential multicollinearity issues")
                    st.write("VIF > 10 indicates serious multicollinearity problems")
                    
                    # Color-code VIF values
                    def color_vif(val):
                        color = 'white'
                        if val > 10:
                            color = 'red'
                        elif val > 5:
                            color = 'yellow'
                        return f'background-color: {color}'
                    
                    st.dataframe(vif_data.style.applymap(color_vif, subset=['VIF']))
                
            except Exception as e:
                st.error(f"Error in regression analysis: {str(e)}")
                st.write("Common issues include:")
                st.write("- Multicollinearity (highly correlated predictors)")
                st.write("- Missing values")
                st.write("- Categorical variables with too many levels")
    
    with tab3:
        st.subheader("Logistic Regression")
        
        # Explain the purpose
        st.write("Logistic regression is used for binary classification problems.")
        
        # Select dependent variable (binary outcome)
        binary_cols = []
        for col in data.columns:
            if data[col].dtype in ['bool', 'object', 'category']:
                if data[col].dropna().nunique() == 2:
                    binary_cols.append(col)
            elif data[col].dtype in ['int64', 'float64']:
                if set(data[col].dropna().unique()).issubset({0, 1}):
                    binary_cols.append(col)
        
        if not binary_cols:
            st.warning("No binary columns found for logistic regression. A binary column has exactly 2 unique values.")
            st.write("Consider using the Data Cleaning page to create a binary column first.")
        else:
            y_var = st.selectbox("Select binary outcome variable (Y):", binary_cols)
            
            # Select independent variables
            x_vars = st.multiselect(
                "Select independent variables (X):",
                [col for col in numeric_cols if col != y_var],
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            # Include categorical variables
            include_categorical = st.checkbox("Include categorical variables", value=False, key="logistic_cat_check")
            cat_vars = []
            
            if include_categorical and categorical_cols:
                cat_vars = st.multiselect(
                    "Select categorical variables:",
                    [col for col in categorical_cols if col != y_var],
                    key="logistic_cat_vars"
                )
            
            # Run analysis when button is clicked
            if st.button("Run Logistic Regression") and x_vars:
                # Create a copy of the data for analysis
                all_vars = [y_var] + x_vars + cat_vars
                reg_data = data[all_vars].copy()
                
                # If binary outcome is not 0/1, convert it
                if reg_data[y_var].dtype not in ['int64', 'float64'] or not set(reg_data[y_var].dropna().unique()).issubset({0, 1}):
                    # Get the unique values
                    unique_vals = reg_data[y_var].dropna().unique()
                    
                    # Map the first value to 0, the second to 1
                    value_map = {unique_vals[0]: 0, unique_vals[1]: 1}
                    reg_data[y_var] = reg_data[y_var].map(value_map)
                    
                    st.write(f"Converted {y_var} to binary: {value_map}")
                
                # Handle categorical variables
                formula = f"{y_var} ~ "
                
                # Add numeric variables
                if x_vars:
                    formula += " + ".join(x_vars)
                
                # Add categorical variables
                if cat_vars:
                    if x_vars:
                        formula += " + "
                    formula += " + ".join([f"C({var})" for var in cat_vars])
                
                # Fit the model
                try:
                    model = sm.Logit(reg_data[y_var], reg_data[x_vars + cat_vars]).fit(disp=0)
                    
                    # Display results
                    st.write("### Logistic Regression Results")
                    
                    # Model metrics
                    ll = model.llf
                    ll_null = model.llnull
                    pseudo_r2 = 1 - (ll / ll_null)
                    
                    st.write(f"**Pseudo R² (McFadden):** {pseudo_r2:.4f}")
                    st.write(f"**Log-Likelihood:** {ll:.4f}")
                    
                    # Interpretation
                    if pseudo_r2 < 0.2:
                        r2_interpretation = "weak"
                    elif pseudo_r2 < 0.4:
                        r2_interpretation = "moderate"
                    else:
                        r2_interpretation = "strong"
                    
                    st.write(f"**Interpretation:** The model has a {r2_interpretation} fit based on McFadden's Pseudo R².")
                    
                    # Display model summary
                    model_summary = model.summary()
                    st.text(str(model_summary))
                    
                    # Odds ratios
                    st.write("### Odds Ratios")
                    st.write("Odds ratios greater than 1 indicate that the variable increases the odds of the outcome.")
                    st.write("Odds ratios less than 1 indicate that the variable decreases the odds of the outcome.")
                    
                    params = model.params
                    conf = model.conf_int()
                    conf['Odds Ratio'] = np.exp(params)
                    conf.columns = ['2.5%', '97.5%', 'Odds Ratio']
                    
                    # Add p-values
                    conf['P-Value'] = model.pvalues
                    
                    # Format for display
                    odds_table = conf[['Odds Ratio', '2.5%', '97.5%', 'P-Value']]
                    
                    # Color-code p-values
                    def color_pval(val):
                        color = 'red'
                        if val < 0.01:
                            color = 'lightgreen'
                        elif val < 0.05:
                            color = 'yellow'
                        return f'background-color: {color}'
                    
                    st.dataframe(odds_table.style.applymap(color_pval, subset=['P-Value']).format({
                        'Odds Ratio': '{:.4f}',
                        '2.5%': '{:.4f}',
                        '97.5%': '{:.4f}',
                        'P-Value': '{:.4f}'
                    }))
                    
                    # Significant predictors
                    sig_predictors = odds_table[odds_table['P-Value'] < 0.05]
                    
                    if not sig_predictors.empty:
                        # Create bar chart of odds ratios
                        fig = px.bar(
                            sig_predictors.reset_index(),
                            y='index',
                            x='Odds Ratio',
                            orientation='h',
                            title='Odds Ratios of Significant Predictors',
                            error_x=dict(
                                type='data',
                                symmetric=False,
                                array=sig_predictors['97.5%'] - sig_predictors['Odds Ratio'],
                                arrayminus=sig_predictors['Odds Ratio'] - sig_predictors['2.5%']
                            ),
                            color='Odds Ratio',
                            color_continuous_scale='RdBu_r',
                            log_x=True
                        )
                        
                        # Add reference line at OR=1
                        fig.add_shape(
                            type="line",
                            line=dict(dash="dash", color="gray"),
                            y0=-0.5,
                            y1=len(sig_predictors) - 0.5,
                            x0=1,
                            x1=1
                        )
                        
                        st.plotly_chart(fig)
                    
                    # Predictions
                    st.write("### Model Evaluation")
                    
                    # Get predictions
                    reg_data['predicted_prob'] = model.predict(reg_data)
                    reg_data['predicted_class'] = (reg_data['predicted_prob'] >= 0.5).astype(int)
                    
                    # Confusion matrix
                    conf_matrix = pd.crosstab(
                        reg_data[y_var],
                        reg_data['predicted_class'],
                        rownames=['Actual'],
                        colnames=['Predicted']
                    )
                    
                    # Make sure both classes are represented in the confusion matrix
                    for i in range(2):
                        for j in range(2):
                            if i not in conf_matrix.index or j not in conf_matrix.columns:
                                if i not in conf_matrix.index:
                                    conf_matrix.loc[i, :] = 0
                                if j not in conf_matrix.columns:
                                    conf_matrix.loc[:, j] = 0
                    
                    # Sort indices and columns
                    conf_matrix = conf_matrix.sort_index().sort_index(axis=1)
                    
                    # Calculate metrics
                    tn = conf_matrix.loc[0, 0] if 0 in conf_matrix.index and 0 in conf_matrix.columns else 0
                    fp = conf_matrix.loc[0, 1] if 0 in conf_matrix.index and 1 in conf_matrix.columns else 0
                    fn = conf_matrix.loc[1, 0] if 1 in conf_matrix.index and 0 in conf_matrix.columns else 0
                    tp = conf_matrix.loc[1, 1] if 1 in conf_matrix.index and 1 in conf_matrix.columns else 0
                    
                    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    # Display confusion matrix
                    st.write("**Confusion Matrix**")
                    
                    # Heatmap of confusion matrix
                    fig = px.imshow(
                        conf_matrix,
                        text_auto=True,
                        title='Confusion Matrix',
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        color_continuous_scale='Blues'
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.2%}")
                    col2.metric("Precision", f"{precision:.2%}")
                    col3.metric("Recall", f"{recall:.2%}")
                    col4.metric("F1 Score", f"{f1:.2%}")
                    
                    # ROC curve
                    st.write("**ROC Curve**")
                    
                    # Calculate ROC curve
                    fpr, tpr, thresholds = stats.roc_curve(reg_data[y_var], reg_data['predicted_prob'])
                    auc = stats.roc_auc_score(reg_data[y_var], reg_data['predicted_prob'])
                    
                    # Create ROC curve
                    fig = px.area(
                        x=fpr, y=tpr,
                        title=f'ROC Curve (AUC = {auc:.4f})',
                        labels=dict(x='False Positive Rate', y='True Positive Rate'),
                        width=700, height=500
                    )
                    
                    # Add diagonal reference line
                    fig.add_shape(
                        type='line',
                        line=dict(dash='dash'),
                        x0=0, x1=1, y0=0, y1=1
                    )
                    
                    st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Error in logistic regression analysis: {str(e)}")
                    st.write("Common issues include:")
                    st.write("- Perfect separation (a predictor perfectly separates the outcome)")
                    st.write("- Multicollinearity (highly correlated predictors)")
                    st.write("- Missing values")
                    st.write("- Too few observations for one of the outcome classes")

def time_series_analysis(data):
    st.header("📅 Time Series Analysis")
    
    # Check for datetime columns
    datetime_cols = []
    
    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            datetime_cols.append(col)
        elif data[col].dtype == 'object':
            # Try to convert to datetime
            try:
                pd.to_datetime(data[col])
                datetime_cols.append(col)
            except:
                pass
    
    if not datetime_cols:
        st.warning("No datetime columns found. Please convert a column to datetime format in the Data Cleaning page.")
        return
    
    # Select datetime column
    date_col = st.selectbox("Select date/time column:", datetime_cols)
    
    # Make sure the column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        try:
            data[date_col] = pd.to_datetime(data[date_col])
        except Exception as e:
            st.error(f"Error converting {date_col} to datetime: {str(e)}")
            return
    
    # Select value column (numeric)
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns available for time series analysis.")
        return
    
    value_col = st.selectbox("Select value column:", numeric_cols)
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Seasonal Decomposition", "Autocorrelation"])
    
    with tab1:
        st.subheader("Trend Analysis")
        
        # Sort data by date
        ts_data = data[[date_col, value_col]].dropna().sort_values(date_col)
        
        # Select time frequency for resampling
        freq_options = {
            'Day': 'D',
            'Week': 'W',
            'Month': 'M',
            'Quarter': 'Q',
            'Year': 'Y'
        }
        
        freq = st.selectbox(
            "Select time frequency for aggregation:",
            list(freq_options.keys())
        )
        
        # Select aggregation method
        agg_method = st.selectbox(
            "Select aggregation method:",
            ['Mean', 'Sum', 'Count', 'Min', 'Max']
        )
        
        agg_func_map = {
            'Mean': 'mean',
            'Sum': 'sum',
            'Count': 'count',
            'Min': 'min',
            'Max': 'max'
        }
        
        # Resample the data
        try:
            ts_resampled = ts_data.set_index(date_col).resample(freq_options[freq])
            ts_resampled = ts_resampled[value_col].agg(agg_func_map[agg_method])
            ts_resampled = ts_resampled.to_frame().reset_index()
            
            # Display resampled data
            st.write(f"### {freq}ly {agg_method} of {value_col}")
            st.dataframe(ts_resampled.head())
            
            # Plot time series
            fig = px.line(
                ts_resampled,
                x=date_col,
                y=value_col,
                title=f"{freq}ly {agg_method} of {value_col} Over Time",
                markers=True
            )
            
            st.plotly_chart(fig)
            
            # Add rolling average
            window_size = st.slider(
                "Rolling average window size:",
                min_value=2,
                max_value=min(30, len(ts_resampled) // 2),
                value=3
            )
            
            # Calculate rolling average
            rolling_data = ts_resampled.copy()
            rolling_data['Rolling Average'] = rolling_data[value_col].rolling(window=window_size).mean()
            
            # Plot with rolling average
            fig = px.line(
                rolling_data,
                x=date_col,
                y=[value_col, 'Rolling Average'],
                title=f"{freq}ly {agg_method} with {window_size}-period Rolling Average",
                markers=True
            )
            
            st.plotly_chart(fig)
            
            # Calculate trend with linear regression
            X = np.arange(len(ts_resampled)).reshape(-1, 1)
            y = ts_resampled[value_col].values
            
            model = sm.OLS(y, sm.add_constant(X)).fit()
            ts_resampled['Trend'] = model.predict(sm.add_constant(X))
            
            # Plot with trend line
            fig = px.line(
                ts_resampled,
                x=date_col,
                y=[value_col, 'Trend'],
                title=f"{freq}ly {agg_method} with Linear Trend",
                markers=True
            )
            
            st.plotly_chart(fig)
            
            # Display trend statistics
            trend_coef = model.params[1]
            trend_pval = model.pvalues[1]
            
            st.write("### Trend Analysis")
            st.write(f"**Trend coefficient:** {trend_coef:.4f} per period")
            
            # Interpret trend
            if trend_pval < 0.05:
                trend_direction = "increasing" if trend_coef > 0 else "decreasing"
                st.write(f"**Interpretation:** There is a statistically significant {trend_direction} trend (p = {trend_pval:.4f}).")
            else:
                st.write(f"**Interpretation:** There is no statistically significant trend (p = {trend_pval:.4f}).")
            
            # Year-over-year comparison if enough data
            if freq in ['Day', 'Week', 'Month'] and ts_resampled[date_col].dt.year.nunique() > 1:
                st.write("### Year-over-Year Comparison")
                
                # Extract year and period
                if freq == 'Day':
                    ts_resampled['Year'] = ts_resampled[date_col].dt.year
                    ts_resampled['Period'] = ts_resampled[date_col].dt.dayofyear
                elif freq == 'Week':
                    ts_resampled['Year'] = ts_resampled[date_col].dt.year
                    ts_resampled['Period'] = ts_resampled[date_col].dt.isocalendar().week
                else:  # Month
                    ts_resampled['Year'] = ts_resampled[date_col].dt.year
                    ts_resampled['Period'] = ts_resampled[date_col].dt.month
                
                # Pivot for year-over-year comparison
                yoy_data = ts_resampled.pivot(index='Period', columns='Year', values=value_col)
                
                # Plot YoY comparison
                fig = px.line(
                    yoy_data,
                    title=f"Year-over-Year Comparison of {value_col}",
                    markers=True
                )
                
                st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error in trend analysis: {str(e)}")
    
    with tab2:
        st.subheader("Seasonal Decomposition")
        
        # This requires a regular time series
        st.write("Seasonal decomposition breaks a time series into trend, seasonal, and residual components.")
        
        # Sort data by date
        ts_data = data[[date_col, value_col]].dropna().sort_values(date_col)
        
        # Check if data has a regular frequency
        regular_freq = False
        ts_index = ts_data[date_col]
        
        # Try to infer frequency
        try:
            # Get differences between consecutive dates
            diff = ts_index.diff().dropna()
            
            # If all differences are the same, we have a regular frequency
            if len(diff.unique()) == 1:
                regular_freq = True
                inferred_freq = pd.infer_freq(ts_index)
            else:
                # Check if most differences are similar
                most_common_diff = diff.value_counts().index[0]
                if (diff == most_common_diff).mean() > 0.8:
                    regular_freq = True
                    inferred_freq = pd.infer_freq(ts_index)
        except:
            pass
        
        if not regular_freq:
            st.warning("Data does not have a regular time frequency. Resampling is required for seasonal decomposition.")
            
            # Select resampling frequency
            freq_options = {
                'Day': 'D',
                'Week': 'W',
                'Month': 'M',
                'Quarter': 'Q',
                'Year': 'Y'
            }
            
            resample_freq = st.selectbox(
                "Select frequency for resampling:",
                list(freq_options.keys()),
                key="seasonal_resample_freq"
            )
            
            # Select aggregation method
            resample_agg = st.selectbox(
                "Select aggregation method:",
                ['Mean', 'Sum', 'Count', 'Min', 'Max'],
                key="seasonal_resample_agg"
            )
            
            agg_func_map = {
                'Mean': 'mean',
                'Sum': 'sum',
                'Count': 'count',
                'Min': 'min',
                'Max': 'max'
            }
            
            # Resample the data
            try:
                ts_resampled = ts_data.set_index(date_col).resample(freq_options[resample_freq])
                ts_resampled = ts_resampled[value_col].agg(agg_func_map[resample_agg])
            except Exception as e:
                st.error(f"Error resampling data: {str(e)}")
                return
        else:
            # Data already has regular frequency
            ts_resampled = ts_data.set_index(date_col)[value_col]
        
        # Select decomposition model
        model_type = st.radio(
            "Select decomposition model:",
            ['Additive', 'Multiplicative']
        )
        
        # Select period for decomposition
        if len(ts_resampled) >= 4:
            # Try to guess a reasonable period
            if resample_freq == 'Day':
                default_period = 7  # Weekly
            elif resample_freq == 'Week':
                default_period = 52  # Annual
            elif resample_freq == 'Month':
                default_period = 12  # Annual
            elif resample_freq == 'Quarter':
                default_period = 4  # Annual
            else:
                default_period = 4
            
            period = st.slider(
                "Select period for seasonal pattern:",
                min_value=2,
                max_value=min(len(ts_resampled) // 2, 52),
                value=default_period
            )
            
            # Run decomposition when button is clicked
            if st.button("Run Seasonal Decomposition"):
                # Check if enough data
                if len(ts_resampled) < 2 * period:
                    st.error(f"Not enough data for decomposition with period {period}. Need at least {2 * period} data points.")
                else:
                    try:
                        # Perform decomposition
                        decomposition = seasonal_decompose(
                            ts_resampled,
                            model='additive' if model_type == 'Additive' else 'multiplicative',
                            period=period
                        )
                        
                        # Plot decomposition
                        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
                        
                        decomposition.observed.plot(ax=ax1)
                        ax1.set_title('Observed')
                        
                        decomposition.trend.plot(ax=ax2)
                        ax2.set_title('Trend')
                        
                        decomposition.seasonal.plot(ax=ax3)
                        ax3.set_title('Seasonal')
                        
                        decomposition.resid.plot(ax=ax4)
                        ax4.set_title('Residual')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Analyze seasonal pattern
                        st.write("### Seasonal Pattern Analysis")
                        
                        # Calculate season strength
                        if model_type == 'Additive':
                            # Variance of seasonality vs. residuals
                            season_var = decomposition.seasonal.var()
                            resid_var = decomposition.resid.var()
                            season_strength = max(0, 1 - (resid_var / (season_var + resid_var)))
                        else:
                            # Variance of log seasonality vs. log residuals
                            season_var = np.log(decomposition.seasonal).var()
                            resid_var = np.log(decomposition.resid).var()
                            season_strength = max(0, 1 - (resid_var / (season_var + resid_var)))
                        
                        st.write(f"**Seasonal strength:** {season_strength:.4f}")
                        
                        if season_strength < 0.3:
                            st.write("**Interpretation:** Weak seasonal pattern")
                        elif season_strength < 0.7:
                            st.write("**Interpretation:** Moderate seasonal pattern")
                        else:
                            st.write("**Interpretation:** Strong seasonal pattern")
                        
                        # Show average seasonal pattern
                        seasonal_data = decomposition.seasonal
                        
                        # If monthly data, show monthly pattern
                        if period == 12:
                            seasonal_pattern = pd.DataFrame(seasonal_data).reset_index()
                            seasonal_pattern['Month'] = seasonal_pattern[date_col].dt.month
                            seasonal_pattern = seasonal_pattern.groupby('Month')[value_col].mean()
                            
                            fig = px.bar(
                                x=seasonal_pattern.index,
                                y=seasonal_pattern.values,
                                title='Average Monthly Seasonal Pattern',
                                labels={'x': 'Month', 'y': 'Seasonal Effect'}
                            )
                            
                            st.plotly_chart(fig)
                        elif period == 4:
                            seasonal_pattern = pd.DataFrame(seasonal_data).reset_index()
                            seasonal_pattern['Quarter'] = seasonal_pattern[date_col].dt.quarter
                            seasonal_pattern = seasonal_pattern.groupby('Quarter')[value_col].mean()
                            
                            fig = px.bar(
                                x=seasonal_pattern.index,
                                y=seasonal_pattern.values,
                                title='Average Quarterly Seasonal Pattern',
                                labels={'x': 'Quarter', 'y': 'Seasonal Effect'}
                            )
                            
                            st.plotly_chart(fig)
                        elif period == 7:
                            seasonal_pattern = pd.DataFrame(seasonal_data).reset_index()
                            seasonal_pattern['Day of Week'] = seasonal_pattern[date_col].dt.dayofweek
                            seasonal_pattern = seasonal_pattern.groupby('Day of Week')[value_col].mean()
                            
                            # Map numbers to day names
                            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            seasonal_pattern.index = [day_names[i] for i in seasonal_pattern.index]
                            
                            fig = px.bar(
                                x=seasonal_pattern.index,
                                y=seasonal_pattern.values,
                                title='Average Day-of-Week Seasonal Pattern',
                                labels={'x': 'Day of Week', 'y': 'Seasonal Effect'}
                            )
                            
                            st.plotly_chart(fig)
                        else:
                            # Generic seasonal pattern
                            seasonal_pattern = pd.DataFrame({
                                'Period': range(1, period + 1),
                                'Seasonal Effect': pd.Series(decomposition.seasonal.iloc[:period].values)
                            })
                            
                            fig = px.bar(
                                seasonal_pattern,
                                x='Period',
                                y='Seasonal Effect',
                                title=f'Seasonal Pattern (Period {period})',
                                labels={'Period': 'Period', 'Seasonal Effect': 'Seasonal Effect'}
                            )
                            
                            st.plotly_chart(fig)
                    
                    except Exception as e:
                        st.error(f"Error in seasonal decomposition: {str(e)}")
                        st.write("Common issues include:")
                        st.write("- Period is too large relative to data length")
                        st.write("- Data has missing values or zeros (for multiplicative model)")
                        st.write("- Data doesn't have a consistent pattern")
        else:
            st.error("Not enough data for seasonal decomposition. Need at least 4 data points.")
    
    with tab3:
        st.subheader("Autocorrelation Analysis")
        
        # Sort data by date
        ts_data = data[[date_col, value_col]].dropna().sort_values(date_col)
        
        # Check if we need to resample to regular frequency
        regular_freq = False
        ts_index = ts_data[date_col]
        
        # Try to infer frequency
        try:
            # Get differences between consecutive dates
            diff = ts_index.diff().dropna()
            
            # If all differences are the same, we have a regular frequency
            if len(diff.unique()) == 1:
                regular_freq = True
                inferred_freq = pd.infer_freq(ts_index)
            else:
                # Check if most differences are similar
                most_common_diff = diff.value_counts().index[0]
                if (diff == most_common_diff).mean() > 0.8:
                    regular_freq = True
                    inferred_freq = pd.infer_freq(ts_index)
        except:
            pass
        
        if not regular_freq:
            st.warning("Data does not have a regular time frequency. Resampling is required for autocorrelation analysis.")
            
            # Select resampling frequency
            freq_options = {
                'Day': 'D',
                'Week': 'W',
                'Month': 'M',
                'Quarter': 'Q',
                'Year': 'Y'
            }
            
            resample_freq = st.selectbox(
                "Select frequency for resampling:",
                list(freq_options.keys()),
                key="acf_resample_freq"
            )
            
            # Select aggregation method
            resample_agg = st.selectbox(
                "Select aggregation method:",
                ['Mean', 'Sum', 'Count', 'Min', 'Max'],
                key="acf_resample_agg"
            )
            
            agg_func_map = {
                'Mean': 'mean',
                'Sum': 'sum',
                'Count': 'count',
                'Min': 'min',
                'Max': 'max'
            }
            
            # Resample the data
            try:
                ts_resampled = ts_data.set_index(date_col).resample(freq_options[resample_freq])
                ts_resampled = ts_resampled[value_col].agg(agg_func_map[resample_agg])
                ts_series = ts_resampled
            except Exception as e:
                st.error(f"Error resampling data: {str(e)}")
                return
        else:
            # Data already has regular frequency
            ts_series = ts_data.set_index(date_col)[value_col]
        
        # Select maximum lag
        max_lag = st.slider(
            "Maximum lag to display:",
            min_value=5,
            max_value=min(40, len(ts_series) // 2),
            value=20
        )
        
        # Run analysis when button is clicked
        if st.button("Run Autocorrelation Analysis"):
            try:
                # Calculate ACF and PACF
                acf_values = acf(ts_series, nlags=max_lag)
                pacf_values = pacf(ts_series, nlags=max_lag)
                
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
                
                # Plot ACF
                ax1.stem(range(len(acf_values)), acf_values)
                ax1.axhline(y=0, linestyle='--', color='gray')
                
                # Add confidence intervals (95%)
                conf_level = 1.96 / np.sqrt(len(ts_series))
                ax1.axhline(y=conf_level, linestyle=':', color='red')
                ax1.axhline(y=-conf_level, linestyle=':', color='red')
                
                ax1.set_title('Autocorrelation Function (ACF)')
                ax1.set_xlabel('Lag')
                ax1.set_ylabel('Correlation')
                
                # Plot PACF
                ax2.stem(range(len(pacf_values)), pacf_values)
                ax2.axhline(y=0, linestyle='--', color='gray')
                
                # Add confidence intervals (95%)
                ax2.axhline(y=conf_level, linestyle=':', color='red')
                ax2.axhline(y=-conf_level, linestyle=':', color='red')
                
                ax2.set_title('Partial Autocorrelation Function (PACF)')
                ax2.set_xlabel('Lag')
                ax2.set_ylabel('Correlation')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Interpretation
                st.write("### Autocorrelation Analysis")
                
                # Identify significant lags
                significant_acf = [i for i in range(1, len(acf_values)) if abs(acf_values[i]) > conf_level]
                significant_pacf = [i for i in range(1, len(pacf_values)) if abs(pacf_values[i]) > conf_level]
                
                if significant_acf:
                    st.write(f"**Significant autocorrelations at lags:** {', '.join(map(str, significant_acf))}")
                else:
                    st.write("**No significant autocorrelations detected.**")
                
                if significant_pacf:
                    st.write(f"**Significant partial autocorrelations at lags:** {', '.join(map(str, significant_pacf))}")
                else:
                    st.write("**No significant partial autocorrelations detected.**")
                
                # Check for seasonality
                if len(significant_acf) > 1:
                    # Look for repeating patterns in the significant lags
                    lag_diffs = np.diff(significant_acf)
                    if len(lag_diffs) > 0 and len(np.unique(lag_diffs)) < len(lag_diffs) / 2:
                        common_diff = stats.mode(lag_diffs).mode[0]
                        st.write(f"**Possible seasonal pattern detected with period {common_diff}.**")
                
                # Stationarity test
                st.write("### Stationarity Test (Augmented Dickey-Fuller)")
                
                # Perform ADF test
                adf_result = adfuller(ts_series.dropna())
                
                st.write(f"**ADF Statistic:** {adf_result[0]:.4f}")
                st.write(f"**p-value:** {adf_result[1]:.4f}")
                
                # Critical values
                st.write("**Critical Values:**")
                for key, value in adf_result[4].items():
                    st.write(f"   {key}: {value:.4f}")
                
                # Interpretation
                if adf_result[1] < 0.05:
                    st.write("**Interpretation:** The time series is stationary (p < 0.05).")
                else:
                    st.write("**Interpretation:** The time series is not stationary (p >= 0.05).")
                    st.write("Non-stationary series may need differencing for further analysis.")
                
                # Plot original vs. differenced series
                if adf_result[1] >= 0.05:
                    st.write("### First Differencing")
                    
                    # Calculate first difference
                    differenced = ts_series.diff().dropna()
                    
                    # Perform ADF test on differenced series
                    diff_adf_result = adfuller(differenced.dropna())
                    
                    # Create plots
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    
                    # Original series
                    ax1.plot(ts_series.index, ts_series.values)
                    ax1.set_title('Original Time Series')
                    
                    # Differenced series
                    ax2.plot(differenced.index, differenced.values)
                    ax2.set_title('First Differenced Series')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show results of ADF test on differenced series
                    st.write(f"**ADF Statistic for differenced series:** {diff_adf_result[0]:.4f}")
                    st.write(f"**p-value for differenced series:** {diff_adf_result[1]:.4f}")
                    
                    if diff_adf_result[1] < 0.05:
                        st.write("**Interpretation:** The differenced series is stationary (p < 0.05).")
                    else:
                        st.write("**Interpretation:** The differenced series is still not stationary (p >= 0.05).")
                        st.write("Higher-order differencing or transformation may be needed.")
            
            except Exception as e:
                st.error(f"Error in autocorrelation analysis: {str(e)}")

    # What's next section
    st.markdown("---")
    st.markdown("## What's Next?")
    st.info("👉 Proceed to the **Predictive Modeling** page to discover intelligent patterns and anomalies in your data.")

if __name__ == "__main__":
    main()