# 🚀 DataInsights Pro

**AI-Powered Data Analytics Platform for Everyone - No Coding Required!**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Google Gemini](https://img.shields.io/badge/Powered%20by-Google%20Gemini%20AI-orange.svg)](https://ai.google.dev/)

<div align="center">
  <img src="https://d3an9kf42ylj3p.cloudfront.net/uploads/2022/08/pg_analyticstools_aug22.jpg" alt="DataInsights Pro Banner" width="800"/>
</div>

## ✨ Overview

**DataInsights Pro** is a comprehensive, AI-powered data analytics platform that democratizes data science. Whether you're a business analyst, researcher, or data enthusiast, our platform enables you to unlock powerful insights from your data without writing a single line of code.

### 🌟 What Makes It Special

- **🤖 AI-Powered Analysis**: Leverages Google's Gemini AI for intelligent data interpretation and insights
- **📊 No-Code Analytics**: Complete data science workflow through an intuitive web interface
- **🔄 End-to-End Pipeline**: From data upload to predictive modeling and report generation
- **💡 Natural Language Queries**: Ask questions about your data in plain English
- **📈 Advanced Visualizations**: Interactive charts and dashboards with Plotly
- **🧹 Smart Data Cleaning**: Automated data quality assessment and cleaning suggestions
- **🔮 Predictive Modeling**: Build ML models without coding knowledge
- **📄 Professional Reports**: Generate comprehensive PDF and Word reports

## 🎯 Key Features

### 📊 **Data Management**
- **Multi-format Support**: CSV, Excel (xlsx/xls), TSV files
- **Smart Upload**: Automatic data type detection and validation
- **Quality Assessment**: Comprehensive data quality scoring (0-100)
- **Missing Value Detection**: Intelligent handling of incomplete data

### 🧹 **Data Cleaning & Preparation**
- **Automated Cleaning**: One-click data cleaning with AI suggestions
- **Outlier Detection**: Multiple algorithms (IQR, Z-Score, Isolation Forest)
- **Data Type Conversion**: Smart type inference and conversion
- **Duplicate Removal**: Intelligent duplicate detection and removal

### 🔍 **Data Exploration**
- **Interactive Filtering**: Real-time data filtering and sorting
- **Statistical Summaries**: Comprehensive descriptive statistics
- **Correlation Analysis**: Automated correlation matrix and heatmaps
- **Distribution Analysis**: Visual distribution analysis for all data types

### 📈 **Advanced Visualizations**
- **20+ Chart Types**: From basic plots to advanced statistical charts
- **Interactive Dashboards**: Plotly-powered interactive visualizations
- **Custom Styling**: Professional themes and color schemes
- **Export Options**: High-quality PNG, SVG, and HTML exports

### 🧠 **AI-Powered Analytics**
- **Natural Language Queries**: Ask questions in plain English
- **Automated Insights**: AI discovers patterns and trends
- **Anomaly Detection**: ML-powered outlier identification
- **Pattern Recognition**: AI identifies hidden relationships

### 📊 **Statistical Analysis**
- **Hypothesis Testing**: t-tests, chi-square, ANOVA, and more
- **Confidence Intervals**: Statistical significance testing
- **Regression Analysis**: Linear and logistic regression
- **Time Series Analysis**: Trend and seasonality detection

### 🔮 **Predictive Modeling**
- **No-Code ML**: Build models without programming
- **Multiple Algorithms**: 
  - **Classification**: Logistic Regression, Random Forest, SVM, Neural Networks
  - **Regression**: Linear, Ridge, Lasso, Gradient Boosting
  - **Clustering**: K-Means, DBSCAN, Hierarchical
- **Model Evaluation**: Comprehensive performance metrics
- **Hyperparameter Tuning**: Automated optimization

### ⏱️ **Time Series Forecasting**
- **Multiple Methods**: ARIMA, Exponential Smoothing, Prophet
- **Trend Analysis**: Automatic trend and seasonality detection
- **Confidence Intervals**: Uncertainty quantification
- **Interactive Plots**: Zoom, pan, and explore forecasts

### 📄 **Report Generation**
- **Professional Reports**: PDF and Word document generation
- **Custom Templates**: Business, academic, and technical formats
- **Automated Insights**: AI-generated conclusions and recommendations
- **Chart Integration**: Embedded visualizations and tables

## 🛠️ Technical Stack

### **Core Technologies**
- **Frontend**: Streamlit (Interactive web interface)
- **Backend**: Python 3.8+
- **AI Engine**: OpenAI API (primary) with Gemini fallback
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Prophet
- **Visualizations**: Plotly, Matplotlib, Seaborn

### **Key Libraries**
```python
# Core Data Science
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0

# AI & NLP
google-generativeai>=0.3.0
openai>=1.40.0
nltk>=3.8

# Visualization
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Web Framework
streamlit>=1.28.0

# Report Generation
reportlab>=3.6.0
python-docx>=0.8.11

# Statistical Analysis
scipy>=1.10.0
statsmodels>=0.14.0
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (recommended) or Google Gemini API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Gaurav-Mishra-17/Data-Insights-Pro.git
cd datainsights-pro
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

# Optional fallback/provider overrides
# Gemini_API_KEY=your_gemini_api_key_here
# OPENAI_MODEL=gpt-4o-mini
# GEMINI_MODEL=gemini-2.0-flash
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
Navigate to `http://localhost:8501` and start analyzing your data!

### 🔑 Getting Your OpenAI API Key (Recommended)

1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an API key
3. Add `OPENAI_API_KEY` to your `.env` file

### 🔑 Getting Your Gemini API Key (Optional Fallback)

1. Visit [Google AI Studio](https://ai.google.dev/)
2. Create a new project or select existing one
3. Generate your API key
4. Add `Gemini_API_KEY` to your `.env` file

## 📝 Usage Guide

### 1. **Upload Your Data**
- Drag and drop your CSV/Excel file
- Review the automatic data quality assessment
- Check data types and missing values

### 2. **Clean Your Data**
- Follow AI-powered cleaning suggestions
- Handle missing values with smart imputation
- Remove duplicates and outliers

### 3. **Explore Patterns**
- Use interactive filtering and sorting
- Generate correlation matrices
- Analyze distributions and relationships

### 4. **Create Visualizations**
- Choose from 20+ chart types
- Customize colors, themes, and layouts
- Export high-quality images

### 5. **Ask AI Questions**
- Type natural language queries
- Get instant insights and explanations
- Generate code for complex analysis

### 6. **Build Predictive Models**
- Select your target variable
- Choose from multiple ML algorithms
- Evaluate model performance automatically

### 7. **Generate Reports**
- Create professional PDF/Word reports
- Include all charts and insights
- Share with stakeholders

## 🎯 Use Cases

### 📈 **Business Analytics**
- Sales performance analysis
- Customer segmentation
- Market trend identification
- Financial forecasting

### 🔬 **Research & Academia**
- Statistical hypothesis testing
- Data exploration and visualization
- Research report generation
- Survey data analysis

### 💼 **Data Science Teams**
- Rapid prototyping
- Non-technical stakeholder demos
- Data quality assessment
- Automated reporting

### 🏥 **Healthcare & Life Sciences**
- Clinical trial data analysis
- Patient outcome prediction
- Epidemiological studies
- Medical research reports

### 📋 **Development Setup**
```bash
# Clone your fork
git clone https://github.com/Gaurav-Mishra-17/Data-Insights-Pro.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 .
black .
```

## 🗺️ Roadmap

### 🔮 **Roadmap**
- [ ] **Database Connectivity** (PostgreSQL, MySQL, MongoDB)
- [ ] **Real-time Data Streaming** (Kafka, WebSocket)
- [ ] **Advanced NLP** (Sentiment analysis, text classification)
- [ ] **Deep Learning Models** (TensorFlow/PyTorch integration)
- [ ] **Collaborative Features** (Team workspaces, sharing)
- [ ] **API Integration** (REST API for programmatic access)
- [ ] **Cloud Deployment** (AWS, GCP, Azure templates)


<div align="center">

### ⭐ **Star this project if you find it useful!** ⭐
```




**Keywords**: data analytics, machine learning, streamlit, python, ai, data science, no-code, business intelligence, visualization, predictive modeling, google gemini ai, automated insights
