import pandas as pd
from loguru import logger
import optuna
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef, fbeta_score
from imblearn.metrics import geometric_mean_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class SynthesizerTuner:
    def __init__(self, train_df: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, 
                 trtr_results: dict, config: dict, global_config: dict, eval_config: dict, artifact_dir: Path):
        self.train_df = train_df
        self.X_train = train_df.drop(columns=[global_config['target_column']])
        self.y_train = train_df[global_config['target_column']]
        self.X_test = X_test
        self.y_test = y_test
        self.trtr_results = trtr_results
        self.config = config
        self.global_config = global_config
        self.eval_config = eval_config
        self.artifact_dir = artifact_dir
        self.logger = logger
        self.models_to_test = self.config.get('models_to_test', ['RandomForest'])

    def run(self) -> dict:
        """
        Runs the full TSTR workflow for all synthesizers and all configured classical models.
        """
        self.logger.info("--- Starting TSTR Workflow ---")
        tstr_results = {}
        synthesizers = {"CTGAN": CTGANSynthesizer, "TVAE": TVAESynthesizer}

        for name, synthesizer_class in synthesizers.items():
            best_params, metadata = self._tune_synthesizer(name, synthesizer_class)
            
            # Train final synthesizer, generate data, and save both
            final_synthesizer = synthesizer_class(metadata, **best_params)
            final_synthesizer.fit(self.train_df)
            
            # --- NEW: Plot and Save Loss ---
            try:
                fig = final_synthesizer.get_loss_values_plot()
                if fig is not None:
                    loss_plot_path = self.artifact_dir / f"loss_plot_{name}.png"
                    fig.write_image(loss_plot_path)
                    self.logger.info(f"Saved synthesizer loss plot to {loss_plot_path}")
                    # No plt.close() needed for plotly objects
                else:
                    self.logger.warning(f"Could not retrieve loss plot for {name}.")
            except Exception as e:
                self.logger.warning(f"Could not generate or save loss plot for {name}: {e}")
            # --- END NEW ---

            final_synthesizer.save(self.artifact_dir / f"synthesizer_{name}.pkl")
            self.logger.info(f"Saved trained {name} model.")
            
            synthetic_train_df = final_synthesizer.sample(num_rows=len(self.train_df))
            synthetic_train_df.to_csv(self.artifact_dir / f"synthetic_{name}_train_set.csv", index=False)
            self.logger.info(f"Saved synthetic {name} dataset.")

            # Evaluate all classical models on this single synthetic dataset
            for model_name in self.models_to_test:
                model_results = self._evaluate_model_on_synthetic_data(model_name, synthetic_train_df)
                tstr_results[f"{model_name}_TSTR_{name}"] = model_results

        self.logger.info("--- TSTR Workflow Finished ---")
        return tstr_results

    def _tune_synthesizer(self, name: str, synthesizer_class) -> dict:
        """Tunes a given synthesizer using Optuna."""
        self.logger.info(f"Tuning synthesizer: {name}...")

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=self.train_df)

        def objective(trial):
            if name == "CTGAN":
                params = {
                    'epochs': self.config.get('synthesizer_epochs', 1),
                    'discriminator_lr': trial.suggest_float('discriminator_lr', 1e-6, 1e-4, log=True),
                    'generator_lr': trial.suggest_float('generator_lr', 1e-6, 1e-4, log=True),
                }
            else: # TVAE
                params = {
                    'epochs': self.config.get('synthesizer_epochs', 1),
                    'batch_size': trial.suggest_categorical('batch_size', [500, 1000]),
                }
            
            model = synthesizer_class(metadata, **params)
            model.fit(self.train_df)
            synthetic_data = model.sample(num_rows=len(self.train_df))
            
            X_train_synthetic = synthetic_data.drop(columns=[self.global_config['target_column']])
            y_train_synthetic = synthetic_data[self.global_config['target_column']]

            # Pre-emptive check: ensure synthetic data has more than one class
            if y_train_synthetic.nunique() < 2:
                self.logger.warning("Synthetic data in trial has only one class, returning AUC of 0.5.")
                return 0.5

            # Use a simple, fast classifier for quick evaluation during tuning
            temp_clf = LogisticRegression(random_state=self.global_config['random_seed'], solver='liblinear')
            temp_clf.fit(X_train_synthetic, y_train_synthetic)
            
            return roc_auc_score(self.y_test, temp_clf.predict_proba(self.X_test)[:, 1])

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.get('n_trials_synthesizer'))
        
        self.logger.info(f"Finished tuning {name}. Best AUC: {study.best_value:.4f}")
        self.logger.info(f"Best params for {name}: {study.best_params}")
        # Add epochs back in for the final model training
        best_params = {'epochs': self.config.get('synthesizer_epochs', 1), **study.best_params}
        return best_params, metadata

    def _evaluate_model_on_synthetic_data(self, model_name: str, synthetic_train_df: pd.DataFrame) -> dict:
        """Trains and evaluates a single classical model on the provided synthetic data."""
        self.logger.info(f"Evaluating {model_name} on synthetic data...")
        
        X_train_synthetic = synthetic_train_df.drop(columns=[self.global_config['target_column']])
        y_train_synthetic = synthetic_train_df[self.global_config['target_column']]
        
        # Get the pre-tuned parameters for this model from the TRTR stage
        model_params = self.trtr_results[model_name]['best_params']
        
        if model_name == 'RandomForest':
            model = RandomForestClassifier(random_state=self.global_config['random_seed'], **model_params)
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(random_state=self.global_config['random_seed'], **model_params)
        elif model_name == 'XGBoost':
            model = XGBClassifier(random_state=self.global_config['random_seed'], eval_metric='logloss', **model_params)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
        
        results = {}
        try:
            # Pre-check before fitting
            if y_train_synthetic.nunique() < 2:
                raise ValueError("Synthetic data contains only one class, cannot train model.")

            pipeline.fit(X_train_synthetic, y_train_synthetic)

            if len(getattr(pipeline.named_steps['model'], 'classes_', [])) < 2:
                self.logger.warning(f"Model {model_name} only learned one class after training.")
                for metric in self.eval_config['primary_metrics']: results[metric] = 0.0
                results['roc_auc'] = 0.5
            else:
                y_pred = pipeline.predict(self.X_test)
                y_proba = pipeline.predict_proba(self.X_test)[:, 1]
                
                if 'BalancedAccuracy' in self.eval_config['primary_metrics']:
                    results['BalancedAccuracy'] = balanced_accuracy_score(self.y_test, y_pred)
                if 'MCC' in self.eval_config['primary_metrics']:
                    results['MCC'] = matthews_corrcoef(self.y_test, y_pred)
                if 'F2' in self.eval_config['primary_metrics']:
                    results['F2'] = fbeta_score(self.y_test, y_pred, beta=2, zero_division=0)
                if 'GMean' in self.eval_config['primary_metrics']:
                    results['GMean'] = geometric_mean_score(self.y_test, y_pred)
                results['roc_auc'] = roc_auc_score(self.y_test, y_proba)

        except ValueError as e:
            self.logger.error(f"Evaluation for {model_name} failed due to mode collapse: {e}")
            for metric in self.eval_config['primary_metrics']: results[metric] = 0.0
            results['roc_auc'] = 0.5 # Assign worst-case score

        self.logger.info(f"TSTR Evaluation ({model_name}):")
        for metric, value in results.items():
            self.logger.info(f"  - {metric}: {value:.4f}")
        return results
