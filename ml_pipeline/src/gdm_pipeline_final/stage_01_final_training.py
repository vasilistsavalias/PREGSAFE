# ml_pipeline/src/gdm_pipeline_final/stage_01_final_training.py
import pandas as pd
from pathlib import Path
import joblib
import logging

from gdm_pipeline_common.config_manager import ConfigManager
from gdm_pipeline_common.components.sdv_wrapper import CTGANWrapper, TVAEWrapper

class FinalModelTrainingStage:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.get_config().final_pipeline
        self.output_path = Path(self.config.artifacts_root)
        self.models_path = self.output_path / "models"
        self.datasets_path = self.output_path / "synthetic_datasets"
        
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.datasets_path.mkdir(parents=True, exist_ok=True)

    def run(self, smoke_test=False):
        logging.info("--- Starting Final Synthesizer Training Stage ---")

        df_original = pd.read_csv(self.config.data_path)
        if smoke_test:
            logging.warning("--- RUNNING IN SMOKE TEST MODE (using first 100 rows & 2 epochs) ---")
            df_original = df_original.head(100)
            # Override epochs for smoke test
            self.config.hyperparameters.ctgan.epochs = 2
            self.config.hyperparameters.tvae.epochs = 2

        # Train and save both synthesizers
        self._train_synthesizer(CTGANWrapper, df_original, self.config.hyperparameters.ctgan, 'ctgan_final.pkl')
        self._train_synthesizer(TVAEWrapper, df_original, self.config.hyperparameters.tvae, 'tvae_final.pkl')

        # Generate datasets and create the balance report
        balance_report_data = []
        
        # Process CTGAN (conditional) -> "optimal"
        ctgan_report = self._generate_dataset(
            original_df=df_original, 
            model_filename='ctgan_final.pkl', 
            isolated_output_filename='isolated_generated_dataset_ctgan.csv',
            final_output_filename='final_ctgan_optimal.csv'
        )
        balance_report_data.append(ctgan_report)
        
        # Process TVAE (unconditional) -> "not_optimal"
        tvae_report = self._generate_dataset(
            original_df=df_original, 
            model_filename='tvae_final.pkl', 
            isolated_output_filename='isolated_generated_dataset_tvae.csv',
            final_output_filename='final_tvae_not_optimal.csv'
        )
        balance_report_data.append(tvae_report)

        # Save the final balance report
        report_df = pd.DataFrame(balance_report_data)
        report_path = self.output_path / "final_balance_report.csv"
        report_df.to_csv(report_path, index=False)
        logging.info(f"Final balance report saved to: {report_path}")

        logging.info("--- Final Synthesizer Training Stage Completed ---")

    def _train_synthesizer(self, wrapper_class, df, params, output_filename):
        synthesizer_name = wrapper_class.__name__.replace("Wrapper", "")
        logging.info(f"--- Training Final {synthesizer_name} Model ---")
        try:
            synthesizer = wrapper_class(params=params.to_dict())
            synthesizer.fit(df)
            output_path = self.models_path / output_filename
            joblib.dump(synthesizer, output_path)
            logging.info(f"Successfully saved trained {synthesizer_name} to: {output_path}")
        except Exception as e:
            logging.error(f"An error occurred during {synthesizer_name} training: {e}", exc_info=True)
            raise

    def _generate_dataset(self, original_df, model_filename, isolated_output_filename, final_output_filename):
        synthesizer_name = model_filename.split('_')[0].upper()
        logging.info(f"--- Processing Dataset Generation for {synthesizer_name} ---")
        try:
            model_path = self.models_path / model_filename
            synthesizer = joblib.load(model_path)

            gdm_counts = original_df['GDM01'].value_counts()
            majority_count = gdm_counts.get(0, 0)
            minority_count = gdm_counts.get(1, 0)

            if majority_count <= minority_count:
                logging.info("Dataset is already balanced. No synthetic samples needed.")
                synthetic_data = pd.DataFrame(columns=original_df.columns)
            else:
                num_to_generate = int(majority_count - minority_count)
                logging.info(f"Generating {num_to_generate} samples with {synthesizer_name}.")

                if synthesizer_name == 'CTGAN':
                    from sdv.sampling import Condition
                    condition = Condition(num_rows=num_to_generate, column_values={'GDM01': 1})
                    synthetic_data = synthesizer.sample(conditions=[condition])
                else: # TVAE generates unconditionally
                    synthetic_data = synthesizer.sample(num_rows=num_to_generate)

            # Save the isolated synthetic data
            isolated_path = self.datasets_path / isolated_output_filename
            synthetic_data.to_csv(isolated_path, index=False)
            logging.info(f"Successfully saved isolated synthetic data to: {isolated_path}")

            # Create and save the final combined dataset
            final_df = pd.concat([original_df, synthetic_data], ignore_index=True)
            final_path = self.datasets_path / final_output_filename
            final_df.to_csv(final_path, index=False)
            logging.info(f"Successfully saved final combined dataset to: {final_path}")

            # Calculate final balance for the report
            final_counts = final_df['GDM01'].value_counts()
            count_0 = final_counts.get(0, 0)
            count_1 = final_counts.get(1, 0)
            total = count_0 + count_1
            
            report = {
                "Dataset": final_output_filename,
                "Count_Class_0": count_0,
                "Count_Class_1": count_1,
                "Percentage_Class_0": round((count_0 / total) * 100, 2) if total > 0 else 0,
                "Percentage_Class_1": round((count_1 / total) * 100, 2) if total > 0 else 0,
                "Total_Samples": total
            }
            return report

        except Exception as e:
            logging.error(f"An error occurred during {synthesizer_name} data generation: {e}", exc_info=True)
            raise

