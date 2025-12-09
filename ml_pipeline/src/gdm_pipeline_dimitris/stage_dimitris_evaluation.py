from gdm_pipeline_common.config_manager import ConfigManager
from gdm_pipeline_common.components.fidelity_analyzer import FidelityAnalyzer
from gdm_pipeline_common.utils.common import save_yaml, generate_balance_report
from loguru import logger
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, fbeta_score, matthews_corrcoef
import json
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport


class StageDimitrisEvaluation:
    def __init__(self, config_manager: ConfigManager, input_dir_override: str = None, output_dir_override: str = None):
        self.config_manager = config_manager
        self.global_params = config_manager.get_config()['global_settings']
        self.eval_config = config_manager.get_config()['final_pipeline']['evaluation']
        
        # The primary input for evaluation is the output of a previous stage.
        if input_dir_override:
            self.input_dir = Path(input_dir_override)
        elif output_dir_override: # Accept output_dir_override as a fallback
            self.input_dir = Path(output_dir_override)
        else:
            # Default to the output of the tuning stage from config
            self.input_dir = Path(config_manager.get_config()['dimitris_pipeline']['tuning']['output_directory'])
        self.logger = logger

    def run(self):
        self.logger.info(">>>>> Starting Stage: Dimitris's Experiment Evaluation <<<<<")
        
        experiment_folders = [f for f in self.input_dir.iterdir() if f.is_dir() and f.name.startswith('Train_set_')]
        if not experiment_folders:
            self.logger.error(f"No experiment folders found in {self.input_dir}")
            return

        # In smoke test mode, we still want to evaluate ALL folders to ensure structure is correct.
        # The speedup comes from the generation stage using fewer epochs.
        if self.config_manager.smoke_test_override:
             self.logger.warning(f"Smoke test mode: Processing ALL {len(experiment_folders)} datasets (Optimization: reduced epochs were used in generation).")

        all_results = []
        for folder in experiment_folders:
            self.logger.info(f"--- Evaluating folder: {folder.name} ---")
            
            # --- Pre-run Verification ---
            synthesizer_path = folder / "CTGAN_synthesizer.pkl"
            if not synthesizer_path.exists():
                self.logger.critical(f"CRITICAL: Synthesizer model not found at {synthesizer_path}. This folder cannot be evaluated. Skipping.")
                continue
            self.logger.success(f"Found synthesizer model at {synthesizer_path}. Proceeding with evaluation.")
            # --------------------------

            try:
                df_original = pd.read_csv(folder / "original_data.csv")
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(df_original)
                
                for synthesizer in ["CTGAN", "TVAE"]:
                    try:
                        self.logger.info(f"--- Evaluating synthesizer: {synthesizer} ---")
                        df_synthetic = pd.read_csv(folder / f"synthetic_data_{synthesizer}.csv")

                        # 1. SDV Full Quality Report
                        self.logger.info("Running SDV Quality Report...")
                        quality_report = evaluate_quality(df_original, df_synthetic, metadata, verbose=False)
                        
                        # Create the new directory structure
                        report_dir = folder / "sdv_reports" / synthesizer
                        report_dir.mkdir(parents=True, exist_ok=True)

                        pkl_report_path = report_dir / f"sdv_quality_report.pkl"
                        quality_report.save(pkl_report_path)
                        self.logger.success(f"SDV Quality Report saved to {pkl_report_path}")
                        
                        # Convert pkl to human-readable yaml
                        yaml_report_path = report_dir / f"sdv_quality_report.yaml"
                        self._save_report_as_yaml(pkl_report_path, yaml_report_path)
                        self.logger.success(f"Human-readable SDV report saved to {yaml_report_path}")

                        self.logger.info(f"Overall Quality Score for {synthesizer}: {quality_report.get_score():.2%}")


                        # 2. Fidelity Analysis (Visual)
                        self.logger.info("Running Fidelity Analysis...")
                        fidelity_output_dir = folder / f"fidelity_plots_{synthesizer}"
                        fidelity_analyzer = FidelityAnalyzer(
                            real_data=df_original.drop(columns=[self.global_params['target_column']]),
                            synthetic_data=df_synthetic.drop(columns=[self.global_params['target_column']]),
                            eval_config=self.eval_config, # Pass the correct config block
                            synthesizer_name=f"{folder.name}_{synthesizer}",
                            output_dir=fidelity_output_dir,
                            logger=self.logger
                        )
                        fidelity_analyzer.run_all_analyses()

                        # 3. Machine Learning Utility (TSTR)
                        self.logger.info("Running Machine Learning Utility (TSTR) Analysis...")
                        tstr_results = self._run_tstr(df_synthetic, df_original)
                        self.logger.info(f"TSTR Results for {synthesizer}: {tstr_results}")
                        
                        result_record = {
                            'fold': folder.name,
                            'synthesizer': synthesizer,
                            'sdv_quality_score': quality_report.get_score(),
                            **tstr_results
                        }
                        all_results.append(result_record)
                    except FileNotFoundError as e:
                        self.logger.warning(f"Skipping synthesizer {synthesizer} in folder {folder.name} due to missing file: {e}")
                        continue

            except Exception as e:
                self.logger.exception(f"Failed to process folder {folder.name}. Skipping.")
                continue
        
        # Save aggregated results
        results_df = pd.DataFrame(all_results)
        results_path = self.input_dir / "aggregated_evaluation_results.csv"
        results_df.to_csv(results_path, index=False)
        self.logger.success(f"Aggregated evaluation results saved to {results_path}")

        # Generate the final balance report
        self.logger.info("Generating final class balance report...")
        generate_balance_report(experiment_folders, self.input_dir, self.global_params['target_column'])

        self.logger.info(">>>>> Stage: Dimitris's Experiment Evaluation Finished <<<<<")

    def _save_report_as_yaml(self, pkl_path: Path, yaml_path: Path):
        """Loads a pickled SDV report and saves its contents as a readable YAML."""
        report = QualityReport.load(pkl_path)
        
        report_dict = {
            'overall_quality_score': report.get_score(),
            'properties': {}
        }

        properties_df = report.get_properties()
        for prop_name in properties_df['Property']:
            prop_data = properties_df.set_index('Property').loc[prop_name]
            report_dict['properties'][prop_name] = {
                'score': prop_data['Score'],
                'details': report.get_details(prop_name).to_dict('records')
            }
            
        save_yaml(report_dict, yaml_path)

    def _run_tstr(self, df_synthetic, df_real):
        """Trains a model on synthetic data and evaluates on real data."""
        target = self.global_params['target_column']
        X_train = df_synthetic.drop(columns=[target])
        y_train = df_synthetic[target]
        X_test = df_real.drop(columns=[target])
        y_test = df_real[target]

        # Using a simple, standard model for evaluation
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=self.global_params['random_seed'], solver='liblinear'))
        ])

        if y_train.nunique() < 2:
            self.logger.warning("Synthetic data has only one class. TSTR cannot be run.")
            return {'roc_auc': 0.5, 'f1_score': 0.0, 'balanced_accuracy': 0.0, 'f2_score': 0.0, 'mcc': 0.0}

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        return {
            'roc_auc': roc_auc_score(y_test, y_proba),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f2_score': fbeta_score(y_test, y_pred, beta=2, zero_division=0),
            'mcc': matthews_corrcoef(y_test, y_pred)
        }
