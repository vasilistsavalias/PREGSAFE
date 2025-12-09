
from gdm_pipeline_common.config_manager import ConfigManager
from gdm_pipeline_common.components.optuna_tuner_tvae import OptunaTunerTVAE
from gdm_pipeline_common.components.optuna_tuner_ctgan import OptunaTunerCTGAN
from gdm_pipeline_common.utils.common import save_yaml
from gdm_pipeline_common.utils.evaluation import convert_numpy_to_native
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer, CTGANSynthesizer
from loguru import logger
from pathlib import Path
import pandas as pd
import glob
from tqdm import tqdm

class StageDimitrisOptunaTuning:
    def __init__(self, config_manager: ConfigManager, output_dir_override: str = None):
        self.config_manager = config_manager
        self.config = config_manager.get_config()['dimitris_pipeline']['tuning']
        self.input_dir = Path(config_manager.get_config()['dimitris_pipeline']['data_path'])
        
        if output_dir_override:
            self.output_dir = Path(output_dir_override)
        else:
            self.output_dir = Path(self.config['output_directory'])
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.synthesizers_to_tune = ["CTGAN", "TVAE"]


    def run(self):
        logger.info(">>>>> Starting Stage: Dimitris's Experiment Optuna Tuning <<<<<")
        
        all_csvs = glob.glob(str(self.input_dir / "*.csv"))
        # Filter to include only Train_set_*.csv and EXCLUDE dataset_filtered.csv if glob picked it up (unlikely with Train_set_*, but safe)
        dataset_paths = sorted([p for p in all_csvs if "Train_set_" in Path(p).name])
        
        # NOTE: In smoke test mode, we still process ALL datasets, but with reduced trials/epochs (handled by config).
        
        logger.info(f"Found {len(dataset_paths)} datasets to tune.")

        all_best_params = {}
        for data_path_str in tqdm(dataset_paths, desc="Tuning Datasets"):
            data_path = Path(data_path_str)
            dataset_name = data_path.stem
            logger.info(f"--- Processing Dataset: {dataset_name} ---")
            
            real_data = pd.read_csv(data_path)
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(real_data)
            
            dataset_output_dir = self.output_dir / dataset_name
            dataset_output_dir.mkdir(exist_ok=True)

            # --- Copy the original data over for the evaluation stage ---
            original_data_path_dest = dataset_output_dir / "original_data.csv"
            real_data.to_csv(original_data_path_dest, index=False)

            all_best_params[dataset_name] = {}

            for synthesizer_name in self.synthesizers_to_tune:
                logger.info(f"--- Tuning for {synthesizer_name} on {dataset_name} ---")

                if synthesizer_name == "TVAE":
                    tuner = OptunaTunerTVAE(real_data, metadata, self.config_manager)
                    synthesizer_class = TVAESynthesizer
                elif synthesizer_name == "CTGAN":
                    tuner = OptunaTunerCTGAN(real_data, metadata, self.config_manager)
                    synthesizer_class = CTGANSynthesizer
                else:
                    logger.warning(f"Unknown synthesizer: {synthesizer_name}. Skipping.")
                    continue

                study_name = f"{dataset_name}_{synthesizer_name}"
                storage_path = self.output_dir / "optuna_studies.db"
                best_params = tuner.run_study(study_name, storage_path)
                all_best_params[dataset_name][synthesizer_name] = best_params

                # --- Generate Optuna Plots ---
                try:
                    import optuna
                    import optuna.visualization.matplotlib as ov
                    import matplotlib.pyplot as plt
                    
                    logger.info(f"Generating Optuna plots for {synthesizer_name}...")
                    # The storage path needs to be a valid SQLite URL
                    # Fix Windows path backslashes if necessary, though usually fine in python
                    storage_url = f"sqlite:///{storage_path.resolve()}"
                    loaded_study = optuna.load_study(study_name=study_name, storage=storage_url)
                    
                    plots_dir = dataset_output_dir / "optuna_plots"
                    plots_dir.mkdir(exist_ok=True)
                    
                    # 1. Optimization History
                    try:
                        ax = ov.plot_optimization_history(loaded_study)
                        # The matplotlib backend returns an Axes object, or Figure. 
                        # check documentation: returns "A matplotlib.axes.Axes object."
                        if hasattr(ax, 'figure'):
                            ax.figure.savefig(plots_dir / f"{synthesizer_name}_optimization_history.png")
                            plt.close(ax.figure)
                    except Exception as e:
                         logger.warning(f"Optimization history plot failed: {e}")

                    # 2. Param Importances
                    try:
                        ax = ov.plot_param_importances(loaded_study)
                        if hasattr(ax, 'figure'):
                            ax.figure.savefig(plots_dir / f"{synthesizer_name}_param_importances.png")
                            plt.close(ax.figure)
                    except Exception:
                        logger.warning(f"Could not generate param importance plot for {synthesizer_name} (maybe too few trials).")

                    # 3. Slice Plot
                    try:
                        ax = ov.plot_slice(loaded_study)
                        if hasattr(ax, 'figure'):
                            ax.figure.savefig(plots_dir / f"{synthesizer_name}_slice_plot.png")
                            plt.close(ax.figure)
                    except Exception as e:
                         logger.warning(f"Slice plot failed: {e}")
                    
                    logger.success(f"Saved Optuna plots to {plots_dir}")
                except Exception as e:
                    logger.error(f"Failed to generate Optuna plots for {synthesizer_name}: {e}")
                
                # Train final model with best params
                logger.info(f"Training final {synthesizer_name} model with best parameters...")
                final_synthesizer = synthesizer_class(metadata, **best_params)
                final_synthesizer.fit(real_data)

                # --- Save the trained synthesizer ---
                synthesizer_save_path = dataset_output_dir / f"{synthesizer_name}_synthesizer.pkl"
                final_synthesizer.save(filepath=synthesizer_save_path)
                logger.success(f"Saved final trained {synthesizer_name} synthesizer to {synthesizer_save_path}")

                # --- Differentiated Sampling Logic ---
                if synthesizer_name == "CTGAN":
                    # --- Conditional Sampling for CTGAN ---
                    target_column = self.config_manager.get_config()['global_settings']['target_column']
                    minority_class_label = 1  # Assuming 1 is the minority class

                    value_counts = real_data[target_column].value_counts()
                    majority_count = value_counts.get(0, 0)
                    minority_count = value_counts.get(1, 0)
                    num_samples_to_generate = int(majority_count - minority_count)

                    if num_samples_to_generate <= 0:
                        logger.warning(f"Dataset {dataset_name} is already balanced. No new samples will be generated for CTGAN.")
                        final_synthetic_data = pd.DataFrame(columns=real_data.columns)
                    else:
                        logger.info(f"Generating {num_samples_to_generate} new samples for the minority class (CTGAN).")
                        from sdv.sampling import Condition
                        condition = Condition(
                            num_rows=num_samples_to_generate,
                            column_values={target_column: minority_class_label}
                        )
                        final_synthetic_data = final_synthesizer.sample_from_conditions(conditions=[condition])
                
                elif synthesizer_name == "TVAE":
                    # --- Unconditional Sampling for TVAE ---
                    num_samples_to_generate = len(real_data)
                    logger.info(f"Generating {num_samples_to_generate} new samples to match original data size (TVAE).")
                    final_synthetic_data = final_synthesizer.sample(num_rows=num_samples_to_generate)
                
                # Save results
                output_path = dataset_output_dir / f"synthetic_data_{synthesizer_name}.csv"
                final_synthetic_data.to_csv(output_path, index=False)
                logger.success(f"Saved optimized data for {synthesizer_name} to {output_path}")

        # Save summary of all best hyperparameters
        summary_path = self.output_dir / "best_hyperparameters_summary.yaml"
        save_yaml(convert_numpy_to_native(all_best_params), summary_path)
        logger.success(f"Hyperparameter summary saved to {summary_path}")
        
        logger.info(">>>>> Stage: Dimitris's Experiment Optuna Tuning Finished <<<<<")
