from gdm_pipeline_common.config_manager import ConfigManager
from gdm_pipeline_common.components.imputation import DataPreprocessor
from gdm_pipeline_common.components.sdv_wrapper import CTGANWrapper, TVAEWrapper
from loguru import logger
from pathlib import Path
import pandas as pd
import os

class StageDimitrisGeneration:
    def __init__(self, config_manager: ConfigManager, output_dir_override: str = None):
        self.config_manager = config_manager
        self.config = config_manager.get_config()['dimitris_pipeline']
        self.global_params = self.config_manager.get_config()['global_settings']
        self.input_dir = Path(self.config['data_path'])
        
        if output_dir_override:
            self.output_dir = Path(output_dir_override)
        else:
            self.output_dir = Path(self.config['tuning']['output_directory'])
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        logger.info(">>>>> Starting Stage: Dimitris's Experiment Data Generation <<<<< ")
        
        # Find all CSV files in the input directory
        try:
            csv_files = list(self.input_dir.glob("*.csv"))
            
            # EXCLUDE dataset_filtered.csv (It belongs to Final Pass)
            csv_files = [f for f in csv_files if f.name != "dataset_filtered.csv"]
            
            if not csv_files:
                logger.error(f"No CSV files found in the specified input directory: {self.input_dir}")
                return
            
            # NOTE: We process ALL found files even in smoke test mode, 
            # but with reduced epochs (handled later).
            
            logger.info(f"Found {len(csv_files)} datasets to process.")
        except Exception as e:
            logger.exception(f"Failed to list CSV files in {self.input_dir}: {e}")
            raise

        # Process each dataset
        for csv_path in csv_files:
            try:
                fold_name = csv_path.stem  # e.g., "Train_set_1"
                logger.info(f"--- Processing dataset: {fold_name} ---")
                
                # Create a dedicated output directory for this fold
                fold_output_dir = self.output_dir / fold_name
                fold_output_dir.mkdir(exist_ok=True)

                # Load data
                df = pd.read_csv(csv_path)
                
                # Drop the unnamed index column as discussed
                if '' in df.columns and df.columns[0] == '':
                    df = df.iloc[:, 1:]
                
                # Save the original data before any transformations
                original_data_path = fold_output_dir / "original_data.csv"
                df.to_csv(original_data_path, index=False)
                logger.info(f"Saved original data for {fold_name} to {original_data_path}")

                # 1. Preprocessing
                logger.info("Initializing and fitting the preprocessor...")
                feature_params = self.config_manager.get_config()['feature_params']
                all_numeric_features = [col for col in df.columns if col != self.global_params['target_column']]
                
                # Filter skewed features to only those present in the current dataframe
                skewed_features_in_df = [f for f in feature_params['skewed_features'] if f in df.columns]
                
                preprocessor = DataPreprocessor(
                    skewed_features=skewed_features_in_df,
                    all_numeric_features=all_numeric_features,
                    random_seed=self.global_params['random_seed']
                )
                transformed_df = preprocessor.fit_transform(df)
                
                # Save the preprocessor for this fold to use for inverse transform
                preprocessor_path = fold_output_dir / "preprocessor.pkl"
                preprocessor.save(preprocessor_path)
                logger.info(f"Preprocessor fitted and saved to {preprocessor_path}")

                # 2. Synthesizer Training and Data Generation
                ctgan_params = self.config_manager.get_config()['final_pipeline']['hyperparameters']['ctgan'].to_dict()
                tvae_params = self.config_manager.get_config()['final_pipeline']['hyperparameters']['tvae'].to_dict()
                
                if self.config_manager.smoke_test_override:
                    # ConfigManager automatically resolves smoke_test/full_run sections
                    smoke_epochs = self.config['generation']['smoke_test']['synthesizer_epochs']
                    logger.warning(f"Smoke test mode active: Overriding epochs to {smoke_epochs}")
                    ctgan_params['epochs'] = smoke_epochs
                    tvae_params['epochs'] = smoke_epochs

                synthesizers = {
                    "CTGAN": CTGANWrapper(ctgan_params),
                    "TVAE": TVAEWrapper(tvae_params)
                }

                for name, synthesizer in synthesizers.items():
                    logger.info(f"--- Training {name} synthesizer for {fold_name} ---")
                    
                    # Train the synthesizer on the *transformed* data
                    synthesizer.fit(transformed_df)
                    
                    # Save the synthesizer model
                    synthesizer_path = fold_output_dir / f"{name}_synthesizer.pkl"
                    synthesizer.save(synthesizer_path)
                    logger.info(f"Synthesizer for {fold_name} saved to {synthesizer_path}")

                    # Add verification step to ensure the model was saved
                    if not os.path.exists(synthesizer_path):
                        error_msg = f"CRITICAL ERROR: Synthesizer model was not saved to disk at {synthesizer_path}"
                        logger.critical(error_msg)
                        raise FileNotFoundError(error_msg)
                    logger.success(f"Verified synthesizer exists on disk: {synthesizer_path}")
                    
                    # Differentiated Sampling Logic
                    if name == "TVAE":
                        # --- Unconditional Sampling for TVAE ---
                        # TVAE does not support conditional sampling well. We generate the same number of rows as the original dataset.
                        num_samples_to_generate_tvae = len(df)
                        logger.info(f"Generating {num_samples_to_generate_tvae} new samples to match original data size (TVAE Unconditional).")
                        synthetic_transformed_df = synthesizer.sample(num_rows=num_samples_to_generate_tvae)
                    else:
                        # --- Conditional Sampling for CTGAN ---
                        # Calculate the number of samples to generate to create a perfectly balanced 50/50 dataset.
                        value_counts = df[target_column].value_counts()
                        majority_count = value_counts.get(0, 0)
                        minority_count = value_counts.get(1, 0)
                        num_samples_to_generate = int(majority_count - minority_count)
                        
                        if num_samples_to_generate <= 0:
                            logger.warning(f"Dataset {fold_name} is already balanced or has no minority samples. Skipping generation.")
                            continue

                        logger.info(f"Implementing 50/50 balancing strategy: Generating {num_samples_to_generate} new minority class samples.")

                        # The condition for the synthesizer needs to use the column name from the transformed dataframe
                        transformed_target_column_name = f"remainder__{target_column}"
                        
                        from sdv.sampling import Condition
                        condition = Condition(
                            num_rows=num_samples_to_generate,
                            column_values={transformed_target_column_name: minority_class_label}
                        )
                        
                        logger.info(f"Generating samples for class '{minority_class_label}' using condition.")
                        synthetic_transformed_df = synthesizer.sample(conditions=[condition])
                    
                    # 3. Inverse Transformation
                    logger.info(f"Applying inverse transform to the {name} synthetic data...")
                    synthetic_original_scale_df = preprocessor.inverse_transform(synthetic_transformed_df)
                    
                    # Ensure column order matches the original dataframe
                    synthetic_original_scale_df = synthetic_original_scale_df[df.columns]

                    # 4. Save the final synthetic data
                    output_filename = f"synthetic_data_{name}.csv"
                    final_output_path = fold_output_dir / output_filename
                    synthetic_original_scale_df.to_csv(final_output_path, index=False)
                    logger.success(f"Successfully generated and saved {name} data for {fold_name} to {final_output_path}")

                    # --- FINAL VERIFICATION STEP ---
                    logger.info(f"--- Verifying Final Balanced Dataset for {name} ---")
                    original_df = pd.read_csv(original_data_path)
                    combined_df = pd.concat([original_df, synthetic_original_scale_df], ignore_index=True)
                    logger.info(f"Final combined dataset class distribution:\n{combined_df[target_column].value_counts().to_string()}")
                    logger.info("--- Final Verification Complete ---")

            except Exception as e:
                logger.exception(f"Failed to process {csv_path}. Skipping to next file.")
                continue

        logger.info(">>>>> Stage: Dimitris's Experiment Data Generation Finished <<<<< ")