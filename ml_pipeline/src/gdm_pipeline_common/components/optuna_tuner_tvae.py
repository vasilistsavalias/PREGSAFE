
import optuna
import pandas as pd
from sdv.single_table import TVAESynthesizer
from gdm_pipeline_common.utils.evaluation import pca_eigen_diff
from loguru import logger

class OptunaTunerTVAE:
    def __init__(self, real_data, metadata, config_manager):
        self.real_data = real_data
        self.metadata = metadata
        self.config = config_manager.get_config()['dimitris_pipeline']['tuning']
        self.is_smoke_test = config_manager.smoke_test_override

    def _objective(self, trial):
        """The objective function for Optuna to minimize."""
        # Define search space based on whether it's a smoke test
        if self.is_smoke_test:
            embedding_dim = trial.suggest_categorical("embedding_dim", [128])
            compress_dims = trial.suggest_categorical("compress_dims", [(256, 256)])
            decompress_dims = trial.suggest_categorical("decompress_dims", [(256, 256)])
            batch_size = trial.suggest_categorical("batch_size", [500])
            epochs = trial.suggest_int("epochs", 1, 2)
        else:
            hp = self.config['hyperparameters_tvae']
            embedding_dim = trial.suggest_categorical("embedding_dim", hp['embedding_dim'])
            compress_dims = trial.suggest_categorical("compress_dims", hp['compress_dims'])
            decompress_dims = trial.suggest_categorical("decompress_dims", hp['decompress_dims'])
            batch_size = trial.suggest_categorical("batch_size", hp['batch_size'])
            epochs = hp['epochs']

        synthesizer = TVAESynthesizer(
            self.metadata,
            embedding_dim=embedding_dim,
            compress_dims=tuple(compress_dims),
            decompress_dims=tuple(decompress_dims),
            batch_size=batch_size,
            epochs=epochs,
            verbose=False
        )
        synthesizer.fit(self.real_data)
        
        synthetic_data = synthesizer.sample(num_rows=len(self.real_data))
        score = pca_eigen_diff(self.real_data, synthetic_data)
        
        return score

    def run_study(self, study_name, storage_path):
        """Runs the Optuna study and returns the best parameters."""
        n_trials = 1 if self.is_smoke_test else self.config['n_trials']
        
        storage = f"sqlite:///{storage_path}"
        logger.info(f"Starting Optuna study for TVAE with {n_trials} trials...")
        logger.info(f"Study Name: {study_name}")
        logger.info(f"Storage: {storage}")

        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.success(f"TVAE Optuna study complete. Best score: {study.best_value:.4f}")
        return study.best_params
