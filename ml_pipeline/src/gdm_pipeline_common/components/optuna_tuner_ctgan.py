
import optuna
from sdv.single_table import CTGANSynthesizer
from gdm_pipeline_common.utils.evaluation import pca_eigen_diff
from loguru import logger

class OptunaTunerCTGAN:
    def __init__(self, real_data, metadata, config_manager):
        self.real_data = real_data
        self.metadata = metadata
        self.config = config_manager.get_config()['dimitris_pipeline']['tuning']
        self.is_smoke_test = config_manager.smoke_test_override

    def _objective(self, trial):
        """The objective function for Optuna to minimize for CTGAN."""
        if self.is_smoke_test:
            embedding_dim = trial.suggest_categorical("embedding_dim", [128])
            generator_dim = trial.suggest_categorical("generator_dim", [(256, 256)])
            discriminator_dim = trial.suggest_categorical("discriminator_dim", [(256, 256)])
            batch_size = trial.suggest_categorical("batch_size", [500])
            epochs = trial.suggest_int("epochs", 1, 2)
        else:
            hp = self.config['hyperparameters_ctgan']
            embedding_dim = trial.suggest_categorical("embedding_dim", hp['embedding_dim'])
            generator_dim = trial.suggest_categorical("generator_dim", hp['generator_dim'])
            discriminator_dim = trial.suggest_categorical("discriminator_dim", hp['discriminator_dim'])
            batch_size = trial.suggest_categorical("batch_size", hp['batch_size'])
            epochs = hp['epochs']

        synthesizer = CTGANSynthesizer(
            self.metadata,
            embedding_dim=embedding_dim,
            generator_dim=tuple(generator_dim),
            discriminator_dim=tuple(discriminator_dim),
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
        logger.info(f"Starting Optuna study for CTGAN with {n_trials} trials...")
        logger.info(f"Study Name: {study_name}")
        logger.info(f"Storage: {storage}")

        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.success(f"CTGAN Optuna study complete. Best score: {study.best_value:.4f}")
        return study.best_params
