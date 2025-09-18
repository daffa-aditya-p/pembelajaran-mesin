"""
training_clean.py
Pipeline Machine Learning untuk Analisis Risiko Kanker Payudara pada Wanita Kuba
Versi yang bersih, ringan, tapi tetap lengkap dan profesional.
"""

import argparse
import logging
import os
import sys
import json
import joblib
import warnings
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

class BreastCancerAnalyzer:
    """Analyzer utama untuk prediksi risiko kanker payudara."""
    
    def __init__(self, output_dir: str = "output", random_state: int = 42):
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.logger = None
        self.setup_logging()
        
        # Model registry
        self.models = {
            'logistic': LogisticRegression(random_state=random_state, max_iter=1000),
            'randomforest': RandomForestClassifier(random_state=random_state, n_estimators=100),
            'gradientboosting': GradientBoostingClassifier(random_state=random_state)
        }
        
        # Results storage
        self.results = {}
        self.best_model = None
        self.preprocessor = None
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"train_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Breast Cancer Risk Analysis Pipeline v3.0.0")
        self.logger.info(f"Log file: {log_file}")
        
    def load_data(self, data_path: str, meta_path: Optional[str] = None) -> pd.DataFrame:
        """Load and prepare dataset."""
        self.logger.info("="*60)
        self.logger.info("MEMULAI ANALISIS RISIKO KANKER PAYUDARA")
        self.logger.info("="*60)
        
        # Load metadata if provided
        if meta_path and os.path.exists(meta_path):
            self.logger.info(f"Loading metadata: {meta_path}")
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = f.read()
                self.logger.info("Dataset metadata loaded successfully")
        
        # Load main dataset
        self.logger.info(f"Loading dataset: {data_path}")
        df = pd.read_csv(data_path)
        self.logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Basic data info
        self.logger.info(f"Columns: {list(df.columns)}")
        self.logger.info(f"Missing values per column:")
        missing_info = df.isnull().sum()
        for col, missing in missing_info.items():
            if missing > 0:
                self.logger.info(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data."""
        self.logger.info("Cleaning data...")
        df_clean = df.copy()
        
        # Handle target variable (cancer column)
        if 'cancer' in df_clean.columns:
            self.logger.info(f"Target variable unique values: {df_clean['cancer'].unique()}")
            # Convert to binary
            df_clean['cancer'] = df_clean['cancer'].map({'Yes': 1, 'No': 0})
            self.logger.info(f"Target distribution: {df_clean['cancer'].value_counts().to_dict()}")
        
        # Remove columns with >80% missing values
        missing_threshold = 0.8
        cols_to_drop = []
        for col in df_clean.columns:
            if col != 'cancer':
                missing_ratio = df_clean[col].isnull().sum() / len(df_clean)
                if missing_ratio > missing_threshold:
                    cols_to_drop.append(col)
        
        if cols_to_drop:
            self.logger.info(f"Dropping columns with >{missing_threshold*100}% missing: {cols_to_drop}")
            df_clean = df_clean.drop(columns=cols_to_drop)
        
        self.logger.info(f"Cleaned dataset shape: {df_clean.shape}")
        return df_clean
    
    def preprocess_features(self, X: pd.DataFrame, y: pd.Series, fit: bool = True) -> pd.DataFrame:
        """Preprocess features."""
        self.logger.info("Preprocessing features...")
        
        if fit:
            # Identify feature types
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            self.logger.info(f"Numeric features: {len(numeric_features)}")
            self.logger.info(f"Categorical features: {len(categorical_features)}")
            
            # Create preprocessing pipeline with imputation
            
            # Numeric pipeline: impute then scale
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # Categorical pipeline: impute then encode
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='drop'
            )
            
            # Fit and transform
            X_processed = self.preprocessor.fit_transform(X)
            
            # Get feature names
            feature_names = self.preprocessor.get_feature_names_out()
            X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
            
        else:
            # Transform only
            X_processed = self.preprocessor.transform(X)
            feature_names = self.preprocessor.get_feature_names_out()
            X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        
        # Feature selection if too many features
        if fit and X_processed.shape[1] > 20:
            self.logger.info("Applying feature selection...")
            selector = SelectKBest(f_classif, k=min(20, X_processed.shape[1]))
            X_processed = selector.fit_transform(X_processed, y)
            selected_features = selector.get_feature_names_out()
            X_processed = pd.DataFrame(X_processed, columns=selected_features, index=X.index)
            self.feature_selector = selector
        
        self.logger.info(f"Final features shape: {X_processed.shape}")
        return X_processed
    
    def handle_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using SMOTE."""
        class_counts = y.value_counts()
        self.logger.info(f"Original class distribution: {class_counts.to_dict()}")
        
        if class_counts.min() / class_counts.max() < 0.5:
            self.logger.info("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            resampled_counts = pd.Series(y_resampled).value_counts()
            self.logger.info(f"Resampled class distribution: {resampled_counts.to_dict()}")
            
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, model_name: str = 'all') -> Dict:
        """Train models and return results."""
        self.logger.info("Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Handle imbalance on training data
        X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train)
        
        results = {}
        models_to_train = [model_name] if model_name != 'all' else list(self.models.keys())
        
        for name in models_to_train:
            if name not in self.models:
                self.logger.warning(f"Unknown model: {name}")
                continue
                
            self.logger.info(f"Training {name}...")
            model = self.models[name]
            
            # Train model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_prob_train = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else y_pred_train
            y_prob_test = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred_test
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5, scoring='roc_auc')
            
            # Calculate metrics
            results[name] = {
                'model': model,
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'train_precision': precision_score(y_train, y_pred_train, zero_division=0),
                'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
                'train_recall': recall_score(y_train, y_pred_train, zero_division=0),
                'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
                'train_f1': f1_score(y_train, y_pred_train, zero_division=0),
                'test_f1': f1_score(y_test, y_pred_test, zero_division=0),
                'train_auc': roc_auc_score(y_train, y_prob_train),
                'test_auc': roc_auc_score(y_test, y_prob_test),
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred_test': y_pred_test,
                'y_prob_test': y_prob_test,
                'confusion_matrix': confusion_matrix(y_test, y_pred_test)
            }
            
            self.logger.info(f"{name} - Test AUC: {results[name]['test_auc']:.4f}, CV AUC: {results[name]['cv_auc_mean']:.4f} ± {results[name]['cv_auc_std']:.4f}")
        
        self.results = results
        return results
    
    def find_best_model(self) -> str:
        """Find best model based on test AUC."""
        if not self.results:
            raise ValueError("No models trained yet")
        
        best_name = max(self.results.keys(), key=lambda x: self.results[x]['test_auc'])
        self.best_model = self.results[best_name]['model']
        self.logger.info(f"Best model: {best_name} (AUC: {self.results[best_name]['test_auc']:.4f})")
        return best_name
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        self.logger.info("Generating visualizations...")
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model Comparison
        if len(self.results) > 1:
            self.logger.info("Creating model comparison plot...")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            models = list(self.results.keys())
            metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            axes = [ax1, ax2, ax3, ax4]
            
            for i, (metric, name, ax) in enumerate(zip(metrics, metric_names, axes)):
                values = [self.results[model][metric] for model in models]
                bars = ax.bar(models, values, alpha=0.7)
                ax.set_title(f'{name} Comparison', fontsize=12, fontweight='bold')
                ax.set_ylabel(name)
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Best Model Performance
        best_name = self.find_best_model()
        best_result = self.results[best_name]
        
        self.logger.info("Creating performance plots...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = best_result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title(f'Confusion Matrix - {best_name}', fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(best_result['y_test'], best_result['y_prob_test'])
        ax2.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {best_result["test_auc"]:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(best_result['y_test'], best_result['y_prob_test'])
        ax3.plot(recall, precision, linewidth=2)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Feature Importance (if available)
        if hasattr(best_result['model'], 'feature_importances_'):
            importances = best_result['model'].feature_importances_
            if hasattr(self, 'feature_selector'):
                feature_names = self.feature_selector.get_feature_names_out()
            else:
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
            
            # Top 10 features
            indices = np.argsort(importances)[::-1][:10]
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]
            
            ax4.barh(range(len(top_importances)), top_importances)
            ax4.set_yticks(range(len(top_importances)))
            ax4.set_yticklabels(top_features)
            ax4.set_xlabel('Importance')
            ax4.set_title('Top 10 Feature Importances', fontweight='bold')
            ax4.invert_yaxis()
        else:
            ax4.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Feature Importance', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(viz_dir / f"best_model_performance_{best_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved to {viz_dir}")
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        self.logger.info("Generating analysis report...")
        
        best_name = self.find_best_model()
        best_result = self.results[best_name]
        
        report = {
            "analysis_info": {
                "timestamp": datetime.now().isoformat(),
                "best_model": best_name,
                "total_models_trained": len(self.results)
            },
            "model_performance": {}
        }
        
        # Add performance metrics for all models
        for name, result in self.results.items():
            report["model_performance"][name] = {
                "test_accuracy": float(result['test_accuracy']),
                "test_precision": float(result['test_precision']),
                "test_recall": float(result['test_recall']),
                "test_f1": float(result['test_f1']),
                "test_auc": float(result['test_auc']),
                "cv_auc_mean": float(result['cv_auc_mean']),
                "cv_auc_std": float(result['cv_auc_std'])
            }
        
        # Save report
        report_file = self.output_dir / "analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create summary table
        summary_file = self.output_dir / "model_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("BREAST CANCER RISK ANALYSIS - MODEL SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Best Model: {best_name}\n")
            f.write(f"Total Models Trained: {len(self.results)}\n\n")
            
            f.write("PERFORMANCE COMPARISON:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}\n")
            f.write("-" * 75 + "\n")
            
            for name, result in self.results.items():
                f.write(f"{name:<15} {result['test_accuracy']:<10.3f} {result['test_precision']:<10.3f} "
                       f"{result['test_recall']:<10.3f} {result['test_f1']:<10.3f} {result['test_auc']:<10.3f}\n")
            
            f.write("\nBEST MODEL DETAILS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Model: {best_name}\n")
            f.write(f"Test Accuracy: {best_result['test_accuracy']:.4f}\n")
            f.write(f"Test AUC: {best_result['test_auc']:.4f}\n")
            f.write(f"Cross-validation AUC: {best_result['cv_auc_mean']:.4f} ± {best_result['cv_auc_std']:.4f}\n")
        
        self.logger.info(f"Report saved to {report_file}")
        self.logger.info(f"Summary saved to {summary_file}")
    
    def save_model(self):
        """Save the best model and preprocessor."""
        if self.best_model is None:
            self.find_best_model()
        
        model_dir = self.output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_dir / "best_model.pkl"
        joblib.dump(self.best_model, model_file)
        
        # Save preprocessor
        if self.preprocessor:
            preprocessor_file = model_dir / "preprocessor.pkl"
            joblib.dump(self.preprocessor, preprocessor_file)
        
        # Save feature selector if exists
        if hasattr(self, 'feature_selector'):
            selector_file = model_dir / "feature_selector.pkl"
            joblib.dump(self.feature_selector, selector_file)
            self.logger.info(f"Feature selector saved to {selector_file}")
        
        self.logger.info(f"Model saved to {model_file}")
        
    def run_complete_analysis(self, data_path: str, meta_path: Optional[str] = None, 
                            model_name: str = 'all') -> Dict:
        """Run complete analysis pipeline."""
        try:
            # Load data
            df = self.load_data(data_path, meta_path)
            
            # Clean data
            df_clean = self.clean_data(df)
            
            # Prepare features and target
            if 'cancer' not in df_clean.columns:
                raise ValueError("Target column 'cancer' not found in dataset")
            
            X = df_clean.drop('cancer', axis=1)
            y = df_clean['cancer']
            
            self.logger.info(f"Features shape: {X.shape}")
            self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            # Preprocess features
            X_processed = self.preprocess_features(X, y, fit=True)
            
            # Train models
            results = self.train_models(X_processed, y, model_name)
            
            # Generate outputs
            self.generate_visualizations()
            self.generate_report()
            self.save_model()
            
            self.logger.info("="*60)
            self.logger.info("ANALISIS SELESAI! ✓")
            self.logger.info("="*60)
            self.logger.info(f"Hasil tersimpan di: {self.output_dir}")
            self.logger.info("Files generated:")
            self.logger.info("- models/best_model.pkl (Model terbaik)")
            self.logger.info("- visualizations/ (Grafik dan visualisasi)")
            self.logger.info("- analysis_report.json (Laporan lengkap)")
            self.logger.info("- model_summary.txt (Ringkasan model)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Breast Cancer Risk Analysis")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--meta", help="Path to metadata file")
    parser.add_argument("--outdir", default="output_clean", help="Output directory")
    parser.add_argument("--model", default="all", choices=['all', 'logistic', 'randomforest', 'gradientboosting'],
                       help="Model to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = BreastCancerAnalyzer(output_dir=args.outdir, random_state=args.seed)
    
    # Run analysis
    results = analyzer.run_complete_analysis(args.data, args.meta, args.model)
    
    return results


if __name__ == "__main__":
    main()