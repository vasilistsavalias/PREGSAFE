from gdm_pipeline_common.config_manager import ConfigManager
from gdm_pipeline_common.logging_config import setup_logging
from gdm_pipeline_dimitris.stage_dimitris_generation import StageDimitrisGeneration
from gdm_pipeline_dimitris.stage_dimitris_optuna_tuning import StageDimitrisOptunaTuning
from gdm_pipeline_dimitris.stage_dimitris_evaluation import StageDimitrisEvaluation
from gdm_pipeline_final.stage_01_final_training import FinalModelTrainingStage
from loguru import logger
from pathlib import Path
import argparse
import os
import pandas as pd

def merge_folds_data(folds_output_dir: Path, target_column: str = 'GDM01'):
    """
    Iterates through each fold directory in the FOLDS_PASS output,
    reads original_data.csv and synthetic_data_{model}.csv,
    merges them, and saves balanced_dataset_{model}.csv.
    """
    logger.info(f"Starting Post-Processing: Merging Fold Data in {folds_output_dir}")
    
    fold_dirs = [d for d in folds_output_dir.iterdir() if d.is_dir() and d.name.startswith("Train_set_")]
    
    if not fold_dirs:
        logger.warning("No fold directories found to process.")
        return

    for fold_dir in fold_dirs:
        original_path = fold_dir / "original_data.csv"
        if not original_path.exists():
            logger.warning(f"Skipping {fold_dir.name}: original_data.csv not found.")
            continue
            
        df_orig = pd.read_csv(original_path)

        for model_name in ["CTGAN", "TVAE"]:
            try:
                synthetic_path = fold_dir / f"synthetic_data_{model_name}.csv"
                output_path = fold_dir / f"balanced_dataset_{model_name}.csv"

                if not synthetic_path.exists():
                    logger.warning(f"Skipping merge for {model_name} in {fold_dir.name}: synthetic data not found.")
                    continue

                df_synth = pd.read_csv(synthetic_path)
                
                # Merge
                df_balanced = pd.concat([df_orig, df_synth], ignore_index=True)
                
                # Save
                df_balanced.to_csv(output_path, index=False)
                
                # Verification Log
                counts = df_balanced[target_column].value_counts()
                logger.success(f"Created {output_path.name} for {fold_dir.name}. Balance: {counts.to_dict()}")

            except Exception as e:
                logger.exception(f"Failed to merge data for {model_name} in {fold_dir.name}: {e}")

def main(args):
    print("--- DEBUG: main_master.py script execution started ---")
    setup_logging(Path("logs/gdm_pipeline_master.log"))
    logger.info("<<<<<<<<<< MASTER PIPELINE STARTING >>>>>>>>>>")
    
    # --- 1. Setup Paths & Config ---
    config_manager = ConfigManager(smoke_test_override=args.smoke_test)
    base_output_dir = Path("outputs/MASTER_RUN")
    
    if args.smoke_test:
        base_output_dir = Path("outputs/MASTER_RUN_SMOKE_TEST")
        logger.warning("--- RUNNING IN SMOKE TEST MODE ---")
    
    folds_pass_dir = base_output_dir / "FOLDS_PASS"
    final_pass_dir = base_output_dir / "FINAL_PASS"
    
    # --- 2. FOLDS PASS (Dimitris Logic) ---
    logger.info("=== STARTING STAGE 1: FOLDS PASS (10 Cross-Validation Folds) ===")
    try:
        # Initialize Dimitris Stage
        # If Tuning is enabled, use the Tuning Stage. Otherwise, use the Generation (Baseline) Stage.
        if args.tune:
            logger.info(">>> MODE: OPTUNA TUNING ENABLED <<<")
            # Note: Tuning stage also generates the final synthetic data using the best model.
            stage = StageDimitrisOptunaTuning(config_manager, output_dir_override=str(folds_pass_dir))
        else:
            logger.info(">>> MODE: BASELINE GENERATION (No Tuning) <<<")
            stage = StageDimitrisGeneration(config_manager, output_dir_override=str(folds_pass_dir))
        
        stage.run()
        
        # Run Evaluation Stage (Fidelity, SDV Reports) for both models
        logger.info(">>> RUNNING EVALUATION (Fidelity & Reports) <<<")
        # StageDimitrisEvaluation usually takes input dir. Here input is the output of the previous stage.
        eval_stage = StageDimitrisEvaluation(config_manager, input_dir_override=str(folds_pass_dir))
        eval_stage.run()
        
        # Post-Processing for Folds
        merge_folds_data(folds_pass_dir, target_column=config_manager.get_config()['global_settings']['target_column'])
        
        logger.success("=== FOLDS PASS COMPLETED ===")
    except Exception as e:
        logger.exception("FOLDS PASS FAILED")
        if not args.smoke_test: # Fail fast unless testing
            raise e

    # --- 3. FINAL PASS (Final Training Logic) ---
    # Only run this if we want the complete dataset processed. 
    # Does tuning apply to the final pass? 
    # Currently `FinalModelTrainingStage` is hardcoded to use params from config.
    # If we want to tune the final model too, we'd need to refactor that class. 
    # For now, assuming Final Pass is always "Train on full data with config params".
    
    logger.info("=== STARTING STAGE 2: FINAL PASS (Complete Dataset) ===")
    try:
        final_config = config_manager.get_config()['final_pipeline']
        final_config['artifacts_root'] = str(final_pass_dir)
        
        final_stage = FinalModelTrainingStage(config_manager)
        final_stage.run(smoke_test=args.smoke_test)
        
        logger.success("=== FINAL PASS COMPLETED ===")
    except Exception as e:
        logger.exception("FINAL PASS FAILED")
        raise e

    logger.info("<<<<<<<<<< MASTER PIPELINE FINISHED SUCCESSFULLY >>>>>>>>>>")
    logger.info(f"Outputs available at: {base_output_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Master GDM Pipeline (Folds + Final).")
    parser.add_argument('--smoke-test', action='store_true', help='Run in smoke test mode (2 epochs, subset of data).')
    parser.add_argument('--tune', action='store_true', help='Run Optuna Tuning for the folds instead of just generation.')
    args = parser.parse_args()
    main(args)
