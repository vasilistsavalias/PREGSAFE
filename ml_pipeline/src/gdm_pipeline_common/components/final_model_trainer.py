import pandas as pd
from loguru import logger
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, fbeta_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from gdm_pipeline.utils.common import save_json
import joblib
from gdm_pipeline.config_manager import ConfigManager
from gdm_pipeline.components.fidelity_analyzer import FidelityAnalyzer
from imblearn.metrics import geometric_mean_score
import matplotlib
matplotlib.use('Agg')

class FinalModelTrainer:
    def __init__(self, dev_df: pd.DataFrame, holdout_df: pd.DataFrame, champion_info: dict, config: dict, global_config: dict, config_manager: ConfigManager, logger):
        self.dev_df = dev_df
        self.holdout_df = holdout_df
        self.champion_info = champion_info
        self.config = config
        self.global_config = global_config
        self.config_manager = config_manager
        self.logger = logger
        self.synthesizer_name = self.champion_info.get('synthesizer', 'TRTR')
        self.cross_val_config = self.config_manager.get_stage_params('cross_validation')
        self.eval_config = self.config_manager.get_config().get('evaluation_params', {})

    def run(self):
        """Runs the final training, generation, and evaluation workflow."""
        self.logger.info("--- Starting Final Model Training Workflow ---")
        
        X_dev = self.dev_df.drop(columns=[self.global_config['target_column']])
        y_dev = self.dev_df[self.global_config['target_column']]
        X_holdout = self.holdout_df.drop(columns=[self.global_config['target_column']])
        y_holdout = self.holdout_df[self.global_config['target_column']]

        if self.champion_info['synthesizer']:
            self.logger.info("Champion is a TSTR model. Training synthesizer...")
            synthetic_data = self._train_synthesizer_and_generate(self.dev_df)
            X_train_final = synthetic_data.drop(columns=[self.global_config['target_column']])
            y_train_final = synthetic_data[self.global_config['target_column']]
            
            # Run fidelity analysis
            fidelity_analyzer = FidelityAnalyzer(
                real_data=X_dev,
                synthetic_data=X_train_final,
                config=self.config,
                eval_config=self.eval_config,
                synthesizer_name=self.synthesizer_name,
                logger=self.logger
            )
            fidelity_analyzer.run_all_analyses()
        else:
            self.logger.info("Champion is a TRTR model. Using real data for final training.")
            X_train_final = X_dev
            y_train_final = y_dev

        self._train_classifier_and_evaluate(X_train_final, y_train_final, X_holdout, y_holdout)
        
        self.logger.info("--- Final Model Training Workflow Finished ---")

    def _train_synthesizer_and_generate(self, dev_df: pd.DataFrame) -> pd.DataFrame:
        """Trains the champion synthesizer on the full dev set and returns the generated data."""
        synth_name = self.champion_info['synthesizer']
        self.logger.info(f"Training champion synthesizer ({synth_name}) on full development set...")

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=dev_df)
        
        epochs = self.cross_val_config.get('synthesizer_epochs', 500)
        self.logger.info(f"Using {epochs} epochs for final synthesizer training.")

        if synth_name == 'CTGAN':
            synthesizer = CTGANSynthesizer(metadata, epochs=epochs, verbose=True)
        elif synth_name == 'TVAE':
            synthesizer = TVAESynthesizer(metadata, epochs=epochs)
        else:
            raise ValueError(f"Unsupported synthesizer: {synth_name}")

        synthesizer.fit(dev_df)
        
        self.logger.info("Generating final synthetic dataset...")
        synthetic_data = synthesizer.sample(num_rows=len(dev_df))
        
        output_path = Path(self.config['root_dir']) / f"final_synthetic_data_{self.synthesizer_name}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_data.to_csv(output_path, index=False)
        self.logger.info(f"Final synthetic dataset saved to {output_path}")
        
        return synthetic_data

    def _train_classifier_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        """Trains the champion classifier and evaluates it on the holdout set."""
        classifier_name = self.champion_info['classifier']
        self.logger.info(f"Training champion classifier ({classifier_name})...")
        
        # CRITICAL FIX: Load the best parameters found during cross-validation
        params = self.champion_info.get('best_params', {})
        if params:
            self.logger.info(f"Using best parameters from CV: {params}")
        else:
            self.logger.warning("No best parameters found in champion_info. Using model defaults.")

        if classifier_name == 'RandomForest':
            model = RandomForestClassifier(random_state=self.global_config['random_seed'], **params)
        elif classifier_name == 'LogisticRegression':
            model = LogisticRegression(random_state=self.global_config['random_seed'], solver='liblinear', **params)
        elif classifier_name == 'XGBoost':
            model = XGBClassifier(random_state=self.global_config['random_seed'], eval_metric='logloss', **params)
        elif classifier_name == 'GaussianNB':
            model = GaussianNB(**params)
        else:
            raise ValueError(f"Unsupported classifier: {classifier_name}")

        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
        
        try:
            # Pre-check before fitting
            if y_train.nunique() < 2:
                raise ValueError("Synthetic data contains only one class, cannot train model.")

            pipeline.fit(X_train, y_train)

            model_path = Path(self.config['root_dir']) / f"final_model_{self.synthesizer_name}.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, model_path)
            self.logger.info(f"Final trained model pipeline saved to {model_path}")

            self.logger.info("Evaluating final model on the untouched holdout set...")
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_dict['roc_auc'] = roc_auc_score(y_test, y_proba)
            report_dict['BalancedAccuracy'] = balanced_accuracy_score(y_test, y_pred)
            report_dict['MCC'] = matthews_corrcoef(y_test, y_pred)
            report_dict['F2'] = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
            report_dict['GMean'] = geometric_mean_score(y_test, y_pred)
        
        except ValueError as e:
            self.logger.error(f"Final evaluation for {classifier_name} failed due to mode collapse: {e}")
            # Create a dummy report for failed runs
            report_dict = {
                'error': str(e), 'roc_auc': 0.5, 'BalancedAccuracy': 0.0, 
                'MCC': 0.0, 'F2': 0.0, 'GMean': 0.0
            }

        self.logger.info(f"\n" + "="*50 + f"\nFINAL CHAMPION REPORT ({self.synthesizer_name})\n" + "="*50)
        self.logger.info(f"  Synthesizer: {self.champion_info['synthesizer']}")
        self.logger.info(f"  Classifier:  {self.champion_info['classifier']}")
        self.logger.info("-" * 50)
        if 'error' not in report_dict:
            self.logger.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
            self.logger.info("-" * 50)
        self.logger.info(f"Final Hold-out Metrics:")
        self.logger.info(f"  - AUC ROC:          {report_dict['roc_auc']:.4f}")
        self.logger.info(f"  - Balanced Accuracy:  {report_dict['BalancedAccuracy']:.4f}")
        self.logger.info(f"  - MCC:                {report_dict['MCC']:.4f}")
        self.logger.info(f"  - F2-Score:           {report_dict['F2']:.4f}")
        self.logger.info(f"  - G-Mean:             {report_dict['GMean']:.4f}")
        self.logger.info("=" * 50)

        report_path = Path(self.config['root_dir']) / f"final_evaluation_report_{self.synthesizer_name}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(report_dict, report_path)
        self.logger.info(f"Final evaluation report saved to {report_path}")
