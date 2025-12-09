import pandas as pd
from loguru import logger
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef, fbeta_score, make_scorer
from imblearn.metrics import geometric_mean_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class ClassicalModelTrainer:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, config: dict, global_config: dict, eval_config: dict):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
        self.global_config = global_config
        self.eval_config = eval_config
        self.logger = logger
        self.models_to_test = self.config.get('models_to_test', ['RandomForest'])

    def run(self) -> dict:
        """Runs the full TRTR workflow for all configured models."""
        self.logger.info("--- Starting TRTR Workflow ---")
        trtr_results = {}
        for model_name in self.models_to_test:
            best_params = self._tune_model(model_name)
            model_results = self._train_and_evaluate_final_model(model_name, best_params)
            trtr_results[model_name] = model_results
        
        self.logger.info("--- TRTR Workflow Finished ---")
        return trtr_results

    def _tune_model(self, model_name: str) -> dict:
        """Tunes the hyperparameters of a given model using Optuna with early stopping."""
        self.logger.info(f"Tuning TRTR model: {model_name}...")

        if model_name == 'GaussianNB':
            self.logger.info("GaussianNB has no hyperparameters to tune.")
            return {}

        def objective(trial):
            if model_name == 'RandomForest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 4, 20),
                }
                model = RandomForestClassifier(random_state=self.global_config['random_seed'], **params)
            elif model_name == 'LogisticRegression':
                params = {'C': trial.suggest_float('C', 1e-4, 1e2, log=True), 'solver': 'liblinear'}
                model = LogisticRegression(random_state=self.global_config['random_seed'], **params)
            elif model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = XGBClassifier(random_state=self.global_config['random_seed'], eval_metric='logloss', **params)
            else:
                raise ValueError(f"Unsupported model for tuning: {model_name}")

            pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.global_config['random_seed'])
            
            metric_mapping = {
                'BalancedAccuracy': 'balanced_accuracy',
                'MCC': 'matthews_corrcoef',
                'F2': make_scorer(fbeta_score, beta=2),
                'GMean': make_scorer(geometric_mean_score)
            }
            scoring_metric_name = self.eval_config['primary_metrics'][0]
            scoring_metric = metric_mapping.get(scoring_metric_name, scoring_metric_name.lower())

            # Implement early stopping for XGBoost
            if model_name == 'XGBoost':
                # XGBoostPruningCallback is not compatible with scikit-learn pipelines.
                # Optuna's MedianPruner will still be effective.
                pass

            score = cross_val_score(pipeline, self.X_train, self.y_train, cv=cv, scoring=scoring_metric).mean()
            
            return score

        pruner = optuna.pruners.MedianPruner(n_startup_trials=self.config.get('pruning_n_startup_trials', 5))
        study = optuna.create_study(direction='maximize', pruner=pruner)
        study.optimize(objective, n_trials=self.config.get('n_trials_classical'))
        
        self.logger.info(f"Finished tuning {model_name}. Best score: {study.best_value:.4f}")
        self.logger.info(f"Best params: {study.best_params}")
        return study.best_params

    def _train_and_evaluate_final_model(self, model_name: str, params: dict) -> dict:
        """Trains the final model and evaluates it using all primary metrics."""
        self.logger.info(f"Training and evaluating final TRTR model: {model_name}...")
        if model_name == 'RandomForest':
            model = RandomForestClassifier(random_state=self.global_config['random_seed'], **params)
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(random_state=self.global_config['random_seed'], **params)
        elif model_name == 'XGBoost':
            model = XGBClassifier(random_state=self.global_config['random_seed'], eval_metric='logloss', **params)
        elif model_name == 'GaussianNB':
            model = GaussianNB(**params)
        else:
            raise ValueError(f"Unsupported model for final training: {model_name}")

        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
        pipeline.fit(self.X_train, self.y_train)

        y_pred = pipeline.predict(self.X_test)
        y_proba = pipeline.predict_proba(self.X_test)[:, 1]

        results = {'best_params': params}
        if 'BalancedAccuracy' in self.eval_config['primary_metrics']:
            results['BalancedAccuracy'] = balanced_accuracy_score(self.y_test, y_pred)
        if 'MCC' in self.eval_config['primary_metrics']:
            results['MCC'] = matthews_corrcoef(self.y_test, y_pred)
        if 'F2' in self.eval_config['primary_metrics']:
            results['F2'] = fbeta_score(self.y_test, y_pred, beta=2, zero_division=0)
        if 'GMean' in self.eval_config['primary_metrics']:
            results['GMean'] = geometric_mean_score(self.y_test, y_pred)
        results['roc_auc'] = roc_auc_score(self.y_test, y_proba)

        self.logger.info(f"TRTR Evaluation ({model_name}):")
        for metric, value in results.items():
            if metric != 'best_params':
                self.logger.info(f"  - {metric}: {value:.4f}")
        return results
