"""
main.py
Command-line interface untuk prediksi risiko kanker payudara
Versi yang disederhanakan untuk compatibility dengan training.py
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import joblib
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class BreastCancerPredictor:
    """Predictor untuk risiko kanker payudara."""
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.model = None
        self.preprocessor = None
        self.feature_selector = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model artifacts."""
        try:
            logger.info(f"Loading model artifacts from {self.model_dir}")
            
            # Load model
            model_path = self.model_dir / "best_model.pkl"
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"Loaded model: {type(self.model).__name__}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load preprocessor
            preprocessor_path = self.model_dir / "preprocessor.pkl"
            if preprocessor_path.exists():
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info(f"Loaded preprocessor")
            else:
                logger.warning("Preprocessor not found, predictions may not work correctly")
            
            # Load feature selector
            selector_path = self.model_dir / "feature_selector.pkl"
            if selector_path.exists():
                self.feature_selector = joblib.load(selector_path)
                logger.info(f"Loaded feature selector")
            else:
                logger.warning("Feature selector not found")
                
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise
    
    def predict_sample(self, sample_data: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict for a single sample.
        
        Args:
            sample_data: Dictionary with feature values
            threshold: Classification threshold
            
        Returns:
            Prediction results
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([sample_data])
            
            # Preprocess if preprocessor available
            if self.preprocessor:
                X_processed = self.preprocessor.transform(df)
                
                # Apply feature selection if available
                if self.feature_selector:
                    X_processed = self.feature_selector.transform(X_processed)
            else:
                X_processed = df
            
            # Predict
            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba(X_processed)[0, 1]
                prediction = 1 if prob >= threshold else 0
            else:
                prediction = self.model.predict(X_processed)[0]
                prob = prediction  # For models without probability
            
            return {
                'prediction': int(prediction),
                'probability': float(prob) if hasattr(self.model, 'predict_proba') else None,
                'risk_level': 'High' if prediction == 1 else 'Low',
                'confidence': f"{prob*100:.1f}%" if hasattr(self.model, 'predict_proba') else "N/A"
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, input_file: str, output_file: str, threshold: float = 0.5):
        """
        Batch prediction from CSV file.
        
        Args:
            input_file: Path to input CSV
            output_file: Path to output JSON
            threshold: Classification threshold
        """
        try:
            logger.info(f"Loading data from {input_file}")
            df = pd.read_csv(input_file)
            
            results = []
            for idx, row in df.iterrows():
                sample_data = row.to_dict()
                try:
                    pred_result = self.predict_sample(sample_data, threshold)
                    pred_result['sample_id'] = idx
                    pred_result['input_data'] = sample_data
                    results.append(pred_result)
                except Exception as e:
                    logger.warning(f"Error predicting sample {idx}: {e}")
                    results.append({
                        'sample_id': idx,
                        'error': str(e),
                        'input_data': sample_data
                    })
            
            # Save results
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Batch predictions saved to {output_file}")
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            'model_type': type(self.model).__name__ if self.model else 'Not loaded',
            'model_dir': str(self.model_dir),
            'has_preprocessor': self.preprocessor is not None,
            'has_feature_selector': self.feature_selector is not None,
            'model_available': self.model is not None
        }
        
        if hasattr(self.model, 'get_params'):
            info['model_params'] = self.model.get_params()
        
        return info
    
    def interactive_mode(self, threshold: float = 0.5):
        """Interactive prediction mode."""
        print("\\n" + "="*60)
        print("BREAST CANCER RISK PREDICTION - INTERACTIVE MODE")
        print("="*60)
        print("Enter patient data for risk assessment.")
        print("Type 'quit' to exit, 'info' for model information\\n")
        
        # Sample features based on the dataset
        sample_features = {
            'age': "Patient age (years)",
            'menarche': "Age at menarche",
            'menopause': "Age at menopause (or current age if premenopausal)",
            'agefirst': "Age at first pregnancy",
            'children': "Number of children",
            'breastfeeding': "Months of breastfeeding",
            'biopsies': "Number of previous biopsies",
            'imc': "Body Mass Index (BMI)",
            'weight': "Weight (kg)",
            'exercise': "Exercise frequency (0=never, 1=sometimes, 2=regular)",
            'alcohol': "Alcohol consumption (0=never, 1=sometimes, 2=regular)",
            'tobacco': "Tobacco use (0=never, 1=former, 2=current)",
        }
        
        while True:
            try:
                command = input("\\nEnter command (predict/info/quit): ").strip().lower()
                
                if command == 'quit':
                    print("Goodbye!")
                    break
                elif command == 'info':
                    info = self.get_model_info()
                    print("\\nModel Information:")
                    for key, value in info.items():
                        print(f"  {key}: {value}")
                elif command == 'predict':
                    print("\\nEnter patient data:")
                    sample_data = {}
                    
                    for feature, description in sample_features.items():
                        while True:
                            try:
                                value = input(f"{feature} ({description}): ").strip()
                                if value:
                                    # Try to convert to numeric
                                    try:
                                        sample_data[feature] = float(value)
                                        break
                                    except ValueError:
                                        sample_data[feature] = value
                                        break
                                else:
                                    break
                            except KeyboardInterrupt:
                                print("\\nOperation cancelled.")
                                return
                    
                    if sample_data:
                        result = self.predict_sample(sample_data, threshold)
                        print("\\n" + "-"*40)
                        print("PREDICTION RESULT:")
                        print("-"*40)
                        print(f"Risk Level: {result['risk_level']}")
                        if result['probability'] is not None:
                            print(f"Probability: {result['confidence']}")
                        print(f"Binary Prediction: {'Positive' if result['prediction'] == 1 else 'Negative'}")
                        print("-"*40)
                    else:
                        print("No data entered.")
                else:
                    print("Unknown command. Use: predict, info, or quit")
                    
            except KeyboardInterrupt:
                print("\\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"Error: {e}")


def create_sample_data() -> Dict[str, Any]:
    """Create sample data for testing."""
    return {
        'id': 9999,
        'age': 45,
        'menarche': 13,
        'menopause': 'No',
        'agefirst': 25,
        'children': 2,
        'breastfeeding': '18 months',
        'nrelbc': 'Mother',
        'biopsies': 0,
        'hyperplasia': 'No',
        'race': 'White',
        'year': 2020,
        'imc': 24.5,
        'weight': 65,
        'exercise': 'Sometimes',
        'alcohol': 'No',
        'tobacco': 'No',
        'allergies': 'No',
        'emotional': 'Joy',
        'depressive': 'No',
        'histologicalclass': '2',
        'birads': '2'
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Breast Cancer Risk Prediction CLI")
    parser.add_argument("--model-dir", required=True, help="Directory with trained model artifacts")
    parser.add_argument("--input", help="Input CSV file or JSON with sample data")
    parser.add_argument("--output", help="Output file for predictions (JSON)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (0.0-1.0)")
    parser.add_argument("--batch", action="store_true", help="Batch prediction mode")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--info", action="store_true", help="Show model information")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--sample", action="store_true", help="Use sample data for testing")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize predictor
        predictor = BreastCancerPredictor(args.model_dir)
        
        # Show model info
        if args.info:
            info = predictor.get_model_info()
            print("\\nModel Information:")
            print("="*40)
            for key, value in info.items():
                print(f"{key}: {value}")
            print("="*40)
            return
        
        # Interactive mode
        if args.interactive:
            predictor.interactive_mode(args.threshold)
            return
        
        # Sample prediction
        if args.sample:
            sample_data = create_sample_data()
            print("\\nUsing sample data:")
            for key, value in sample_data.items():
                print(f"  {key}: {value}")
            
            result = predictor.predict_sample(sample_data, args.threshold)
            print("\\nPrediction Result:")
            print("="*40)
            for key, value in result.items():
                print(f"{key}: {value}")
            print("="*40)
            return
        
        # Batch mode
        if args.batch and args.input and args.output:
            predictor.predict_batch(args.input, args.output, args.threshold)
            return
        
        # Single prediction from JSON file
        if args.input and args.input.endswith('.json'):
            with open(args.input, 'r') as f:
                sample_data = json.load(f)
            
            result = predictor.predict_sample(sample_data, args.threshold)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Prediction saved to {args.output}")
            else:
                print("\\nPrediction Result:")
                print(json.dumps(result, indent=2))
            return
        
        # Default: show help
        parser.print_help()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()