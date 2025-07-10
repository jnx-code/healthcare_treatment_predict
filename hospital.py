import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HospitalPricePrediction:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.target_column = 'treatment_cost'
        
    def generate_sample_data(self, n_samples=5000):
        """Generate realistic hospital treatment data"""
        np.random.seed(42)
        
        treatments = {
            'Emergency Surgery': {'base_cost': 15000, 'complexity': 'high'},
            'Cardiac Surgery': {'base_cost': 25000, 'complexity': 'high'},
            'Orthopedic Surgery': {'base_cost': 12000, 'complexity': 'medium'},
            'General Surgery': {'base_cost': 8000, 'complexity': 'medium'},
            'Diagnostic Imaging': {'base_cost': 1500, 'complexity': 'low'},
            'Physical Therapy': {'base_cost': 200, 'complexity': 'low'},
            'Chemotherapy': {'base_cost': 5000, 'complexity': 'medium'},
            'Radiation Therapy': {'base_cost': 3000, 'complexity': 'medium'},
            'ICU Stay': {'base_cost': 2000, 'complexity': 'high'},
            'Regular Checkup': {'base_cost': 300, 'complexity': 'low'}
        }
        
        data = []
        for _ in range(n_samples):
            age = np.random.randint(18, 90)
            gender = np.random.choice(['Male', 'Female'])
            
            treatment = np.random.choice(list(treatments.keys()))
            treatment_info = treatments[treatment]
            
            departments = ['Cardiology', 'Orthopedics', 'Emergency', 'General Surgery', 
                         'Oncology', 'Radiology', 'Physical Medicine']
            department = np.random.choice(departments)
            
            insurance_types = ['Private', 'Government', 'Self-Pay', 'Medicare', 'Medicaid']
            insurance = np.random.choice(insurance_types)
            
            if treatment_info['complexity'] == 'high':
                length_of_stay = np.random.randint(3, 21)
            elif treatment_info['complexity'] == 'medium':
                length_of_stay = np.random.randint(1, 7)
            else:
                length_of_stay = np.random.randint(1, 3)
            
            severity = np.random.randint(1, 11)
            
            complications = np.random.choice([0, 1], p=[0.8, 0.2])
            
            room_types = ['General Ward', 'Semi-Private', 'Private', 'ICU']
            room_type = np.random.choice(room_types)
            
            base_cost = treatment_info['base_cost']
            
            age_factor = 1 + (age - 50) * 0.01 if age > 50 else 1
            
            stay_factor = 1 + (length_of_stay - 1) * 0.1
            
            severity_factor = 1 + (severity - 1) * 0.05
            
            complication_factor = 1.5 if complications else 1
            
            room_factors = {'General Ward': 1, 'Semi-Private': 1.2, 'Private': 1.5, 'ICU': 2.0}
            room_factor = room_factors[room_type]
            
            insurance_factors = {'Private': 1.1, 'Government': 0.8, 'Self-Pay': 1.3, 
                               'Medicare': 0.9, 'Medicaid': 0.7}
            insurance_factor = insurance_factors[insurance]
            
            final_cost = (base_cost * age_factor * stay_factor * severity_factor * 
                         complication_factor * room_factor * insurance_factor)
            
            final_cost *= np.random.uniform(0.8, 1.2)
            final_cost = round(final_cost, 2)
            
            data.append({
                'patient_age': age,
                'gender': gender,
                'treatment_type': treatment,
                'department': department,
                'insurance_type': insurance,
                'length_of_stay': length_of_stay,
                'severity_score': severity,
                'complications': complications,
                'room_type': room_type,
                'treatment_cost': final_cost
            })
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        """Preprocess the data for machine learning"""
        processed_df = df.copy()
        
        categorical_cols = ['gender', 'treatment_type', 'department', 'insurance_type', 'room_type']
        
        for col in categorical_cols:
            if col == 'gender':
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col])
                self.encoders[col] = le
            else:
                dummies = pd.get_dummies(processed_df[col], prefix=col)
                processed_df = pd.concat([processed_df, dummies], axis=1)
                processed_df = processed_df.drop(col, axis=1)
                self.encoders[col] = list(dummies.columns)
        
        self.feature_columns = [col for col in processed_df.columns if col != self.target_column]
        
        return processed_df
    
    def train_models(self, X_train, y_train):
        """Train multiple machine learning models"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['feature_scaler'] = scaler
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0)
        }
        
        for name, model in models.items():
            print(f"Training {name}...")
            if name in ['Support Vector Regression', 'Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)
            self.models[name] = model
        
        print("All models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        X_test_scaled = self.scalers['feature_scaler'].transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            if name in ['Support Vector Regression', 'Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
        
        return results
    
    def predict_price(self, patient_data):
        """Predict treatment cost for a new patient"""
        best_model = self.models['Random Forest']
        
        processed_data = self.preprocess_single_input(patient_data)
        
        prediction = best_model.predict([processed_data])[0]
        
        return round(prediction, 2)
    
    def preprocess_single_input(self, patient_data):
        """Preprocess a single patient input for prediction"""
        processed_input = np.zeros(len(self.feature_columns))
        
        feature_mapping = {
            'patient_age': patient_data.get('patient_age', 45),
            'length_of_stay': patient_data.get('length_of_stay', 3),
            'severity_score': patient_data.get('severity_score', 5),
            'complications': patient_data.get('complications', 0)
        }
        
        if patient_data.get('gender') == 'Male':
            feature_mapping['gender'] = 1
        else:
            feature_mapping['gender'] = 0
        
        for feature, value in feature_mapping.items():
            if feature in self.feature_columns:
                idx = self.feature_columns.index(feature)
                processed_input[idx] = value
        
        categorical_features = ['treatment_type', 'department', 'insurance_type', 'room_type']
        
        for cat_feature in categorical_features:
            if cat_feature in patient_data:
                feature_name = f"{cat_feature}_{patient_data[cat_feature]}"
                if feature_name in self.feature_columns:
                    idx = self.feature_columns.index(feature_name)
                    processed_input[idx] = 1
        
        return processed_input
    
    def plot_results(self, results):
        """Plot model comparison results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        models = list(results.keys())
        metrics = ['RMSE', 'MAE', 'R2', 'MSE']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [results[model][metric] for model in models]
            
            bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 
                                               'gold', 'lightpink', 'lightsalmon'])
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance(self):
        """Display feature importance for tree-based models"""
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            importance = rf_model.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance_df.head(15), x='Importance', y='Feature')
            plt.title('Top 15 Feature Importance (Random Forest)')
            plt.tight_layout()
            plt.show()
            
            return feature_importance_df
        
        return None

def main():
    """Main execution function"""
    predictor = HospitalPricePrediction()
    
    print("Generating sample hospital data...")
    df = predictor.generate_sample_data(n_samples=5000)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nDataset statistics:")
    print(df.describe())
    
    print("\nPreprocessing data...")
    processed_df = predictor.preprocess_data(df)
    
    X = processed_df.drop(predictor.target_column, axis=1)
    y = processed_df[predictor.target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    predictor.train_models(X_train, y_train)
    
    print("\nEvaluating models...")
    results = predictor.evaluate_models(X_test, y_test)
    
    print("\nModel Performance Results:")
    print("-" * 80)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    predictor.plot_results(results)
    
    print("\nFeature Importance Analysis:")
    feature_imp = predictor.feature_importance()
    if feature_imp is not None:
        print(feature_imp.head(10))
    
    print("\nExample Prediction:")
    sample_patient = {
        'patient_age': 65,
        'gender': 'Male',
        'treatment_type': 'Cardiac Surgery',
        'department': 'Cardiology',
        'insurance_type': 'Private',
        'length_of_stay': 7,
        'severity_score': 8,
        'complications': 1,
        'room_type': 'ICU'
    }
    
    predicted_cost = predictor.predict_price(sample_patient)
    print(f"Predicted treatment cost: ${predicted_cost:,.2f}")
    
    print("\nPatient details:")
    for key, value in sample_patient.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()