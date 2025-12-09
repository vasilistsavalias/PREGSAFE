import yaml
from pathlib import Path
import pandas as pd
from loguru import logger
import pickle
import json
from box import ConfigBox
import numpy as np

def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Loads a dataset from a CSV or Excel file based on the file extension.
    """
    logger.info(f"Attempting to load dataset from: {file_path}")
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Dataset file not found at path: {file_path}")
    
    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        df.columns = df.columns.str.strip()
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}", exc_info=True)
        raise

def load_yaml(file_path: Path) -> ConfigBox:
    """Loads a YAML file and returns it as a ConfigBox."""
    try:
        with open(file_path, 'r') as f:
            content = yaml.safe_load(f)
        logger.info(f"Successfully loaded YAML from: {file_path}")
        return ConfigBox(content)
    except Exception as e:
        logger.error(f"Failed to load YAML from {file_path}. Error: {e}")
        raise

def save_yaml(data: dict, file_path: Path):
    """Saves a dictionary to a YAML file, converting numpy types to native Python types."""
    
    # Custom Dumper to handle numpy types for clean YAML output
    class NumpySafeDumper(yaml.SafeDumper):
        def represent_data(self, data):
            if isinstance(data, np.integer):
                return self.represent_int(int(data))
            if isinstance(data, np.floating):
                return self.represent_float(float(data))
            return super().represent_data(data)

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(data, f, Dumper=NumpySafeDumper, default_flow_style=False, sort_keys=False)
        logger.info(f"Successfully saved YAML to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save YAML to {file_path}. Error: {e}")
        raise

def save_pickle(obj: object, file_path: Path):
    """Saves a Python object to a pickle file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Successfully saved pickle object to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save pickle to {file_path}. Error: {e}")
        raise

def save_json(data: dict, file_path: Path):
    """Saves a dictionary to a JSON file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Successfully saved JSON to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}. Error: {e}")
        raise


def generate_balance_report(dataset_folders: list, output_root: Path, target_column: str):
    """Generates a CSV report detailing the class balance of the final combined datasets."""
    balance_data = []

    for folder in dataset_folders:
        try:
            original_df = pd.read_csv(folder / "original_data.csv")
            # Assumes CTGAN is the synthesizer used for balancing. This might need adjustment if other synthesizers are used.
            synthetic_df = pd.read_csv(folder / f"synthetic_data_CTGAN.csv")
            combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
            
            counts = combined_df[target_column].value_counts()
            percentages = combined_df[target_column].value_counts(normalize=True) * 100
            
            balance_data.append({
                'Dataset': folder.name,
                'Count_Class_0': counts.get(0, 0),
                'Count_Class_1': counts.get(1, 0),
                'Percentage_Class_0': percentages.get(0, 0),
                'Percentage_Class_1': percentages.get(1, 0),
                'Total_Samples': len(combined_df)
            })
        except FileNotFoundError:
            logger.warning(f"Could not find data for {folder.name} to generate balance report. Skipping.")
            continue
    
    if not balance_data:
        logger.warning("No data found to generate a balance report.")
        return

    report_df = pd.DataFrame(balance_data)
    report_path = output_root / "final_balance_report.csv"
    report_df.to_csv(report_path, index=False)
    logger.success(f"Final balance report saved to {report_path}")
    logger.info(f"\n--- Final Balance Report ---\n{report_df.to_string()}\n--------------------------\n")
