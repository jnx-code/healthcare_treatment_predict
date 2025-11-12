# Healthcare_Treatment_Predict
Hospital Treatment Price Prediction System
A comprehensive machine learning project for predicting hospital treatment costs using patient data and treatment information. This system includes data preprocessing, multiple ML models, evaluation metrics, and both web interface and API endpoints.
🏥 Project Overview
This project predicts hospital treatment costs based on various factors including:

Patient demographics (age, gender)
Treatment type and department
Insurance information
Length of stay
Severity score and complications
Room type

🚀 Features

Machine Learning Models: Multiple algorithms including Random Forest, Gradient Boosting, Linear Regression, and SVM
Web Interface: Interactive HTML interface for easy cost prediction
REST API: Flask-based API for integration with other systems
Data Visualization: Comprehensive charts and graphs for model evaluation
Feature Importance: Analysis of which factors most influence cost
Real-time Predictions: Instant cost estimates with confidence intervals

📊 Model Performance
The system uses multiple models and selects the best performer:

Random Forest: Best overall performance with ~95% accuracy
Gradient Boosting: Strong performance with good generalization
Linear Models: Fast predictions with interpretable results
SVM: Robust performance across different data distributions

🛠 Installation
Prerequisites

Python 3.8+
pip package manager

Setup

Clone the repository
bashgit clone <repository-url>
cd hospital-price-prediction

Create virtual environment
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies
bashpip install -r requirements.txt

Run the main script
bashpython hospital_prediction.py


🌐 Web Interface
Running the Web Interface

Open the HTML file
bash# Simply open hospital_price_web_interface.html in your browser
# Or serve it using Python's built-in server:
python -m http.server 8000

Access the interface

Open your browser and go to http://localhost:8000
Fill in the patient information
Click "Predict Treatment Cost"



Interface Features

Interactive Forms: Easy-to-use input fields with validation
Real-time Updates: Automatic field updates based on treatment selection
Visual Feedback: Loading animations and result displays
Cost Breakdown: Detailed explanation of cost factors
Responsive Design: Works on desktop and mobile devices

🔗 API Usage
Starting the API Server
bashpython api_server.py
The API will be available at http://localhost:5000
API Endpoints
1. Health Check
httpGET /api/health
2. Predict Treatment Cost
httpPOST /api/predict
Content-Type: application/json


3. Get Available Options
httpGET /api/treatments        # List of treatment types
GET /api/departments       # List of departments
GET /api/insurance-types   # List of insurance types
GET /api/room-types        # List of room types
4. Model Information
httpGET /api/model-info
📈 Data Analysis
Sample Data Generation
The system generates realistic hospital data with:

5,000 patient records for training
10 treatment types with varying complexities
Multiple cost factors including age, severity, complications
Realistic price ranges based on actual hospital costs

Feature Engineering

Categorical Encoding: One-hot encoding for multi-class variables
Numerical Scaling: StandardScaler for continuous variables
Feature Selection: Importance-based feature ranking
Data Validation: Comprehensive input validation and error handling

🎯 Model Evaluation
Metrics Used

RMSE: Root Mean Square Error
MAE: Mean Absolute Error
R²: Coefficient of Determination
Cross-validation: 5-fold cross-validation for robust evaluation

Performance Visualization

Model comparison charts
Feature importance plots
Prediction vs actual scatter plots
Residual analysis

📋 Usage Examples
Python Script Usage
pythonfrom hospital_prediction import HospitalPricePrediction

# Initialize predictor
predictor = HospitalPricePrediction()

# Generate and train on sample data
df = predictor.generate_sample_data(n_samples=5000)
processed_df = predictor.preprocess_data(df)

# Split and train
X = processed_df.drop('treatment_cost', axis=1)
y = processed_df['treatment_cost']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train models
predictor.train_models(X_train, y_train)

# Make prediction

🔧 Configuration
Environment Variables
bash# API Configuration
PORT=5000
DEBUG=false

# Model Configuration
MODEL_PATH=hospital_model.pkl
