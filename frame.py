from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import logging

from hospital import HospitalPricePrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

predictor = None
model_loaded = False

def load_model():
    """Load the trained model"""
    global predictor, model_loaded
    
    try:
        if os.path.exists('hospital_model.pkl'):
            logger.info("Loading saved model...")
            predictor = joblib.load('hospital_model.pkl')
            model_loaded = True
            logger.info("Model loaded successfully!")
        else:
            logger.info("No saved model found. Training new model...")
            predictor = HospitalPricePrediction()
            
            df = predictor.generate_sample_data(n_samples=5000)
            processed_df = predictor.preprocess_data(df)
            
            X = processed_df.drop(predictor.target_column, axis=1)
            y = processed_df[predictor.target_column]
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            predictor.train_models(X_train, y_train)
            
            joblib.dump(predictor, 'hospital_model.pkl')
            logger.info("Model trained and saved successfully!")
            
            model_loaded = True
            
    except Exception as e:
        logger.error(f"Error loading/training model: {str(e)}")
        model_loaded = False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict_price():
    """Predict treatment cost based on patient data"""
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        required_fields = [
            'patient_age', 'gender', 'treatment_type', 'department',
            'insurance_type', 'length_of_stay', 'severity_score',
            'complications', 'room_type'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        try:
            age = int(data['patient_age'])
            if age < 18 or age > 100:
                return jsonify({'error': 'Age must be between 18 and 100'}), 400
            
            length_of_stay = int(data['length_of_stay'])
            if length_of_stay < 1 or length_of_stay > 30:
                return jsonify({'error': 'Length of stay must be between 1 and 30 days'}), 400
            
            severity_score = int(data['severity_score'])
            if severity_score < 1 or severity_score > 10:
                return jsonify({'error': 'Severity score must be between 1 and 10'}), 400
            
            complications = int(data['complications'])
            if complications not in [0, 1]:
                return jsonify({'error': 'Complications must be 0 or 1'}), 400
                
        except ValueError:
            return jsonify({'error': 'Invalid data types for numeric fields'}), 400
        
        predicted_cost = predictor.predict_price(data)
        
        base_cost = get_base_cost(data['treatment_type'])
        confidence_lower = predicted_cost * 0.85
        confidence_upper = predicted_cost * 1.15
        
        breakdown = calculate_cost_breakdown(data, predicted_cost)
        
        response = {
            'predicted_cost': round(predicted_cost, 2),
            'confidence_interval': {
                'lower': round(confidence_lower, 2),
                'upper': round(confidence_upper, 2)
            },
            'base_cost': base_cost,
            'breakdown': breakdown,
            'patient_info': {
                'age': age,
                'gender': data['gender'],
                'treatment': data['treatment_type'],
                'department': data['department'],
                'insurance': data['insurance_type'],
                'room_type': data['room_type'],
                'length_of_stay': length_of_stay,
                'severity': severity_score,
                'complications': bool(complications)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made: ${predicted_cost:.2f} for {data['treatment_type']}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/treatments', methods=['GET'])
def get_treatments():
    """Get list of available treatments"""
    treatments = [
        'Emergency Surgery', 'Cardiac Surgery', 'Orthopedic Surgery',
        'General Surgery', 'Diagnostic Imaging', 'Physical Therapy',
        'Chemotherapy', 'Radiation Therapy', 'ICU Stay', 'Regular Checkup'
    ]
    return jsonify({'treatments': treatments})

@app.route('/api/departments', methods=['GET'])
def get_departments():
    """Get list of available departments"""
    departments = [
        'Cardiology', 'Orthopedics', 'Emergency', 'General Surgery',
        'Oncology', 'Radiology', 'Physical Medicine'
    ]
    return jsonify({'departments': departments})

@app.route('/api/insurance-types', methods=['GET'])
def get_insurance_types():
    """Get list of available insurance types"""
    insurance_types = ['Private', 'Government', 'Self-Pay', 'Medicare', 'Medicaid']
    return jsonify({'insurance_types': insurance_types})

@app.route('/api/room-types', methods=['GET'])
def get_room_types():
    """Get list of available room types"""
    room_types = ['General Ward', 'Semi-Private', 'Private', 'ICU']
    return jsonify({'room_types': room_types})

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about the trained model"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        model_info = {
            'model_type': 'Random Forest Regressor',
            'training_samples': 5000,
            'features': len(predictor.feature_columns) if predictor.feature_columns else 0,
            'available_models': list(predictor.models.keys()) if predictor.models else [],
            'last_trained': datetime.now().isoformat(),  # In real app, store this
            'version': '1.0.0'
        }
        
        return jsonify(model_info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def get_base_cost(treatment_type):
    """Get base cost for a treatment type"""
    base_costs = {
        'Emergency Surgery': 15000,
        'Cardiac Surgery': 25000,
        'Orthopedic Surgery': 12000,
        'General Surgery': 8000,
        'Diagnostic Imaging': 1500,
        'Physical Therapy': 200,
        'Chemotherapy': 5000,
        'Radiation Therapy': 3000,
        'ICU Stay': 2000,
        'Regular Checkup': 300
    }
    return base_costs.get(treatment_type, 5000)

def calculate_cost_breakdown(data, predicted_cost):
    """Calculate cost breakdown for explanation"""
    base_cost = get_base_cost(data['treatment_type'])
    
    age_factor = 1.2 if int(data['patient_age']) > 65 else 1.1 if int(data['patient_age']) > 50 else 1.0
    stay_factor = 1 + (int(data['length_of_stay']) - 1) * 0.15
    severity_factor = 1 + (int(data['severity_score']) - 1) * 0.08
    complication_factor = 1.5 if int(data['complications']) else 1.0
    
    room_factors = {'General Ward': 1.0, 'Semi-Private': 1.2, 'Private': 1.5, 'ICU': 2.0}
    room_factor = room_factors.get(data['room_type'], 1.0)
    
    insurance_factors = {'Private': 1.1, 'Government': 0.8, 'Self-Pay': 1.3, 'Medicare': 0.9, 'Medicaid': 0.7}
    insurance_factor = insurance_factors.get(data['insurance_type'], 1.0)
    
    return {
        'base_cost': base_cost,
        'age_adjustment': round((age_factor - 1) * 100, 1),
        'stay_adjustment': round((stay_factor - 1) * 100, 1),
        'severity_adjustment': round((severity_factor - 1) * 100, 1),
        'complication_adjustment': round((complication_factor - 1) * 100, 1),
        'room_adjustment': round((room_factor - 1) * 100, 1),
        'insurance_adjustment': round((insurance_factor - 1) * 100, 1)
    }

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':

    load_model()
 
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Hospital Price Prediction API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)