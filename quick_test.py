"""
quick_test.py - Enhanced End-to-End Pipeline Verification
Comprehensive testing script for the breast cancer risk analysis pipeline.
Tests training, inference, and evaluation components.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineTester:
    """Comprehensive pipeline testing class."""
    
    def __init__(self, base_dir: str = None):
        """Initialize tester with base directory."""
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.temp_dir = None
        self.test_results = {
            "training": {"status": "pending", "details": {}},
            "inference": {"status": "pending", "details": {}},
            "artifacts": {"status": "pending", "details": {}},
            "integration": {"status": "pending", "details": {}}
        }
        
    def setup_test_environment(self):
        """Setup temporary test environment."""
        logger.info("Setting up test environment")
        self.temp_dir = Path(tempfile.mkdtemp(prefix="breast_cancer_test_"))
        logger.info(f"Test directory: {self.temp_dir}")
        
        # Create test data directory
        test_data_dir = self.temp_dir / "data"
        test_data_dir.mkdir()
        
        return test_data_dir
    
    def create_sample_data(self, data_dir: Path) -> Dict[str, Path]:
        """Create sample data for testing."""
        logger.info("Creating sample data")
        
        # Create sample CSV data
        np.random.seed(42)
        n_samples = 100
        
        sample_data = {
            'age': np.random.randint(25, 70, n_samples),
            'imc': np.random.normal(25, 5, n_samples),
            'menarche': np.random.randint(10, 16, n_samples),
            'menopause': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'children': np.random.randint(0, 5, n_samples),
            'agefirst': np.random.randint(18, 35, n_samples),
            'breastfeeding': np.random.randint(0, 24, n_samples),
            'familyhist': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'biopsies': np.random.randint(0, 3, n_samples),
            'location': np.random.choice(['havana', 'santiago', 'camaguey'], n_samples),
            'year': np.random.randint(2015, 2023, n_samples),
            'cancer': np.random.choice(['No', 'Yes'], n_samples, p=[0.7, 0.3])
        }
        
        # Ensure some logical consistency
        for i in range(n_samples):
            if sample_data['children'][i] == 0:
                sample_data['agefirst'][i] = 0
                sample_data['breastfeeding'][i] = 0
            elif sample_data['agefirst'][i] > sample_data['age'][i]:
                sample_data['agefirst'][i] = sample_data['age'][i] - 5
        
        df = pd.DataFrame(sample_data)
        
        # Save CSV
        csv_path = data_dir / "sample_data.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Created sample CSV: {csv_path}")
        
        # Create metadata file
        metadata_text = """Dataset Features:
Age: Patient age in years
IMC: Body Mass Index (BMI)
Menarche: Age at first menstruation
Menopause: Menopause status (0=No, 1=Yes)
Children: Number of children
Agefirst: Age at first birth
Breastfeeding: Breastfeeding duration in months
Familyhist: Family history of cancer (0=No, 1=Yes)
Biopsies: Number of biopsies performed
Location: Geographic location
Year: Year of assessment
Cancer: Target variable (yes/no)

Note:
The values of "No" and "0" (zero) have the same meaning.
The values of "Yes" and "1" (one) have the same meaning.
"""
        
        metadata_path = data_dir / "metadata.txt"
        with open(metadata_path, 'w') as f:
            f.write(metadata_text)
        logger.info(f"Created metadata file: {metadata_path}")
        
        return {
            "csv": csv_path,
            "metadata": metadata_path,
            "dataframe": df
        }
    
    def test_training_pipeline(self, data_paths: Dict[str, Path]) -> bool:
        """Test the training pipeline."""
        logger.info("Testing training pipeline")
        
        try:
            # Setup output directory
            output_dir = self.temp_dir / "training_output"
            output_dir.mkdir()
            
            # Prepare training command
            training_script = self.base_dir / "training.py"
            if not training_script.exists():
                raise FileNotFoundError(f"Training script not found: {training_script}")
            
            cmd = [
                sys.executable, str(training_script),
                "--data", str(data_paths["csv"]),
                "--meta", str(data_paths["metadata"]),
                "--outdir", str(output_dir),
                "--model", "randomforest",  # Use simpler model for testing
                "--seed", "42"  # Set random seed
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Training failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                self.test_results["training"]["status"] = "failed"
                self.test_results["training"]["details"] = {
                    "error": result.stderr,
                    "return_code": result.returncode
                }
                return False
            
            # Check if artifacts were created
            artifacts_dir = output_dir / "artifacts"
            expected_artifacts = [
                "model.joblib",
                "scaler.joblib", 
                "selected_features.joblib",
                "config.json"
            ]
            
            missing_artifacts = []
            for artifact in expected_artifacts:
                if not (artifacts_dir / artifact).exists():
                    missing_artifacts.append(artifact)
            
            if missing_artifacts:
                logger.warning(f"Missing artifacts: {missing_artifacts}")
            
            # Check if reports were generated
            reports_dir = output_dir / "reports"
            if reports_dir.exists():
                report_files = list(reports_dir.glob("*.md"))
                logger.info(f"Generated {len(report_files)} report files")
            
            self.test_results["training"]["status"] = "passed"
            self.test_results["training"]["details"] = {
                "output_dir": str(output_dir),
                "missing_artifacts": missing_artifacts,
                "artifacts_found": len(expected_artifacts) - len(missing_artifacts)
            }
            
            logger.info("Training pipeline test passed")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Training pipeline timed out")
            self.test_results["training"]["status"] = "timeout"
            return False
        except Exception as e:
            logger.error(f"Training pipeline test failed: {e}")
            self.test_results["training"]["status"] = "error"
            self.test_results["training"]["details"] = {"error": str(e)}
            return False
    
    def test_inference_pipeline(self, data_paths: Dict[str, Path]) -> bool:
        """Test the inference pipeline."""
        logger.info("Testing inference pipeline")
        
        try:
            # First check if training artifacts exist
            training_output = self.temp_dir / "training_output"
            if not training_output.exists():
                logger.error("Training output directory not found. Run training test first.")
                self.test_results["inference"]["status"] = "failed"
                return False
            
            # Prepare inference script
            inference_script = self.base_dir / "main.py"
            if not inference_script.exists():
                raise FileNotFoundError(f"Inference script not found: {inference_script}")
            
            # Test 1: Model info
            logger.info("Testing model info")
            cmd_info = [
                sys.executable, str(inference_script),
                "--model-dir", str(training_output),
                "--info"
            ]
            
            result_info = subprocess.run(cmd_info, capture_output=True, text=True, timeout=60)
            
            if result_info.returncode != 0:
                logger.error(f"Model info test failed: {result_info.stderr}")
                self.test_results["inference"]["status"] = "failed"
                return False
            
            logger.info("Model info test passed")
            
            # Test 2: Sample prediction
            logger.info("Testing sample prediction")
            cmd_sample = [
                sys.executable, str(inference_script),
                "--model-dir", str(training_output),
                "--sample"
            ]
            
            result_sample = subprocess.run(cmd_sample, capture_output=True, text=True, timeout=60)
            
            if result_sample.returncode != 0:
                logger.error(f"Sample prediction test failed: {result_sample.stderr}")
                self.test_results["inference"]["status"] = "failed"
                return False
            
            logger.info("Sample prediction test passed")
            
            # Test 3: Batch prediction
            logger.info("Testing batch prediction")
            output_file = self.temp_dir / "predictions.json"
            
            cmd_batch = [
                sys.executable, str(inference_script),
                "--model-dir", str(training_output),
                "--input", str(data_paths["csv"]),
                "--output", str(output_file)
            ]
            
            result_batch = subprocess.run(cmd_batch, capture_output=True, text=True, timeout=60)
            
            if result_batch.returncode != 0:
                logger.error(f"Batch prediction test failed: {result_batch.stderr}")
                self.test_results["inference"]["status"] = "failed"
                return False
            
            # Check if output file was created
            if not output_file.exists():
                logger.error("Batch prediction output file not created")
                self.test_results["inference"]["status"] = "failed"
                return False
            
            # Validate output format
            with open(output_file, 'r') as f:
                predictions = json.load(f)
            
            required_keys = ["predictions", "probabilities", "risk_scores", "num_samples"]
            missing_keys = [key for key in required_keys if key not in predictions]
            
            if missing_keys:
                logger.error(f"Missing keys in prediction output: {missing_keys}")
                self.test_results["inference"]["status"] = "failed"
                return False
            
            logger.info("Batch prediction test passed")
            
            self.test_results["inference"]["status"] = "passed"
            self.test_results["inference"]["details"] = {
                "predictions_file": str(output_file),
                "num_predictions": predictions["num_samples"]
            }
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Inference pipeline timed out")
            self.test_results["inference"]["status"] = "timeout"
            return False
        except Exception as e:
            logger.error(f"Inference pipeline test failed: {e}")
            self.test_results["inference"]["status"] = "error"
            self.test_results["inference"]["details"] = {"error": str(e)}
            return False
    
    def test_artifacts_integrity(self) -> bool:
        """Test the integrity of generated artifacts."""
        logger.info("Testing artifacts integrity")
        
        try:
            training_output = self.temp_dir / "training_output"
            artifacts_dir = training_output / "artifacts"
            
            if not artifacts_dir.exists():
                logger.error("Artifacts directory not found")
                self.test_results["artifacts"]["status"] = "failed"
                return False
            
            # Test model loading
            import joblib
            
            model_path = artifacts_dir / "model.joblib"
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    logger.info(f"Successfully loaded model: {type(model).__name__}")
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    self.test_results["artifacts"]["status"] = "failed"
                    return False
            
            # Test scaler loading
            scaler_path = artifacts_dir / "scaler.joblib"
            if scaler_path.exists():
                try:
                    scaler = joblib.load(scaler_path)
                    logger.info(f"Successfully loaded scaler: {type(scaler).__name__}")
                except Exception as e:
                    logger.error(f"Failed to load scaler: {e}")
                    self.test_results["artifacts"]["status"] = "failed"
                    return False
            
            # Test config loading
            config_path = artifacts_dir / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    logger.info(f"Successfully loaded config with {len(config)} parameters")
                except Exception as e:
                    logger.error(f"Failed to load config: {e}")
                    self.test_results["artifacts"]["status"] = "failed"
                    return False
            
            # Check visualization files
            plots_dir = training_output / "svgs"
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.svg"))
                logger.info(f"Found {len(plot_files)} visualization files")
            
            # Check reports
            reports_dir = training_output / "reports"
            if reports_dir.exists():
                report_files = list(reports_dir.glob("*.md"))
                logger.info(f"Found {len(report_files)} report files")
            
            self.test_results["artifacts"]["status"] = "passed"
            self.test_results["artifacts"]["details"] = {
                "artifacts_loaded": True,
                "visualizations_count": len(plot_files) if 'plot_files' in locals() else 0,
                "reports_count": len(report_files) if 'report_files' in locals() else 0
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Artifacts integrity test failed: {e}")
            self.test_results["artifacts"]["status"] = "error"
            self.test_results["artifacts"]["details"] = {"error": str(e)}
            return False
    
    def test_integration(self, data_paths: Dict[str, Path]) -> bool:
        """Test end-to-end integration."""
        logger.info("Testing end-to-end integration")
        
        try:
            # Load original data
            df = data_paths["dataframe"]
            
            # Create a small test dataset
            test_df = df.head(10).copy()
            test_df = test_df.drop(columns=['cancer'])  # Remove target for prediction
            
            test_csv = self.temp_dir / "test_input.csv"
            test_df.to_csv(test_csv, index=False)
            
            # Run prediction
            training_output = self.temp_dir / "training_output"
            inference_script = self.base_dir / "main.py"
            output_file = self.temp_dir / "integration_test.json"
            
            cmd = [
                sys.executable, str(inference_script),
                "--model-dir", str(training_output),
                "--input", str(test_csv),
                "--output", str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"Integration test failed: {result.stderr}")
                self.test_results["integration"]["status"] = "failed"
                return False
            
            # Validate results
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            if results["num_samples"] != len(test_df):
                logger.error(f"Expected {len(test_df)} predictions, got {results['num_samples']}")
                self.test_results["integration"]["status"] = "failed"
                return False
            
            # Check that all predictions are valid
            predictions = results["predictions"]
            probabilities = results["probabilities"]["positive"]
            
            if len(predictions) != len(probabilities):
                logger.error("Mismatch between predictions and probabilities length")
                self.test_results["integration"]["status"] = "failed"
                return False
            
            # Check value ranges
            if not all(p in [0, 1] for p in predictions):
                logger.error("Invalid prediction values (should be 0 or 1)")
                self.test_results["integration"]["status"] = "failed"
                return False
            
            if not all(0 <= p <= 1 for p in probabilities):
                logger.error("Invalid probability values (should be between 0 and 1)")
                self.test_results["integration"]["status"] = "failed"
                return False
            
            logger.info("End-to-end integration test passed")
            
            self.test_results["integration"]["status"] = "passed"
            self.test_results["integration"]["details"] = {
                "test_samples": len(test_df),
                "predictions_valid": True,
                "probabilities_valid": True
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            self.test_results["integration"]["status"] = "error"
            self.test_results["integration"]["details"] = {"error": str(e)}
            return False
    
    def cleanup(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            logger.info(f"Cleaning up test directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        logger.info("Starting comprehensive pipeline testing")
        
        try:
            # Setup
            data_dir = self.setup_test_environment()
            data_paths = self.create_sample_data(data_dir)
            
            # Run tests in sequence
            tests = [
                ("Training Pipeline", self.test_training_pipeline, data_paths),
                ("Inference Pipeline", self.test_inference_pipeline, data_paths),
                ("Artifacts Integrity", self.test_artifacts_integrity, None),
                ("Integration Test", self.test_integration, data_paths)
            ]
            
            for test_name, test_func, test_args in tests:
                logger.info(f"Running {test_name}")
                try:
                    if test_args:
                        success = test_func(test_args)
                    else:
                        success = test_func()
                    
                    if success:
                        logger.info(f"✓ {test_name} PASSED")
                    else:
                        logger.error(f"✗ {test_name} FAILED")
                        
                except Exception as e:
                    logger.error(f"✗ {test_name} ERROR: {e}")
            
            # Calculate overall status
            passed_tests = sum(1 for test in self.test_results.values() if test["status"] == "passed")
            total_tests = len(self.test_results)
            
            overall_status = "passed" if passed_tests == total_tests else "failed"
            
            results = {
                "overall_status": overall_status,
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "test_details": self.test_results,
                "test_directory": str(self.temp_dir)
            }
            
            return results
            
        finally:
            # Cleanup
            # self.cleanup()  # Comment out for debugging
            pass


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Pipeline Testing")
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory with scripts")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files for debugging")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run tests
    tester = PipelineTester(args.base_dir)
    
    try:
        results = tester.run_all_tests()
        
        # Print summary
        print("\n" + "=" * 60)
        print("PIPELINE TEST SUMMARY")
        print("=" * 60)
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"Tests Passed: {results['passed_tests']}/{results['total_tests']}")
        
        print("\nDetailed Results:")
        for test_name, test_result in results['test_details'].items():
            status_icon = "✓" if test_result["status"] == "passed" else "✗"
            print(f"  {status_icon} {test_name.replace('_', ' ').title()}: {test_result['status'].upper()}")
            
            if test_result["status"] in ["failed", "error"] and "error" in test_result.get("details", {}):
                print(f"    Error: {test_result['details']['error']}")
        
        if args.keep_temp:
            print(f"\nTest files preserved in: {results['test_directory']}")
        
        print("=" * 60)
        
        # Exit with appropriate code
        sys.exit(0 if results['overall_status'] == 'passed' else 1)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if not args.keep_temp:
            tester.cleanup()


if __name__ == "__main__":
    main()
import os
import joblib
import pandas as pd
from training import main as train_main
from main import main as infer_main

def run_quick_test():
    # Buat sample kecil dari dataset
    sample_path = "sample_data.csv"
    df = pd.read_csv("Dataset 1- Breast cancer risk factors in Cuban women Dataset (1).csv")
    df_sample = df.sample(n=50, random_state=42)
    df_sample.to_csv(sample_path, index=False)
    # Training singkat
    os.system(f"python training.py --data {sample_path} --meta 'Dataset 1 - Breast cancer risk factors in Cuban women Dataset Description Detail.txt' --outdir output_test --model xgboost --cv 2 --trials 2 --seed 42 --verbose")
    # Load model dan prediksi
    os.system(f"python main.py --model output_test/model_best.joblib --data {sample_path} --outdir output_test_run --predict --svgs output_test_run/svgs --eval")
    print("Quick test selesai. Cek output_test dan output_test_run.")

if __name__ == "__main__":
    run_quick_test()
