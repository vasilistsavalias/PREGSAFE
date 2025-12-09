import pandas as pd
from loguru import logger

class DataValidator:
    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df
        self.config = config

    def perform_initial_inspection(self):
        """
        Performs and logs high-level data inspection.
        """
        logger.info("--- Performing Initial Data Inspection ---")
        logger.info(f"Dataset Shape: {self.df.shape}")
        
        # In a script, we log info instead of printing it.
        # The .info() method prints to stdout, so we'll capture it.
        import io
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        logger.debug(f"Dataset Info:\n{buffer.getvalue()}")

        logger.debug(f"Descriptive Statistics:\n{self.df.describe().to_string()}")

        missing_values = self.df.isnull().sum()
        missing_values_report = missing_values[missing_values > 0]
        if not missing_values_report.empty:
            logger.info(f"Found {len(missing_values_report)} columns with missing values.")
            logger.debug(f"Missing values report:\n{missing_values_report.to_string()}")
        else:
            logger.info("No missing values found in the dataset.")

        num_duplicates = self.df.duplicated().sum()
        logger.info(f"Found {num_duplicates} duplicated rows.")
        logger.info("--- Initial Data Inspection Finished ---")

    def remove_leakage_features(self) -> pd.DataFrame:
        """
        Removes features that are known to cause target leakage.
        """
        logger.info("--- Removing Leakage Features ---")
        leakage_features = self.config.get('leakage_features_to_exclude', [])
        features_to_drop = [feat for feat in leakage_features if feat in self.df.columns]

        if features_to_drop:
            logger.warning(f"Removing {len(features_to_drop)} leakage features: {features_to_drop}")
            df_cleaned = self.df.drop(columns=features_to_drop)
            logger.info(f"Shape after removing leakage features: {df_cleaned.shape}")
            return df_cleaned
        else:
            logger.info("No target leakage features to remove.")
            return self.df.copy()

    def handle_clinical_outliers(self) -> pd.DataFrame:
        """
        Replaces clinically implausible values with NaN to be imputed later.
        """
        logger.info("--- Handling Clinical Outliers ---")
        df_out = self.df.copy()
        # Define plausible ranges
        ranges = {
            'MA': (12, 60),
            'Wt pre': (30, 200),
            'Ht': (130, 210)
        }
        
        for col, (min_val, max_val) in ranges.items():
            if col in df_out.columns:
                original_nulls = df_out[col].isnull().sum()
                # Important: Use .loc for assignment to avoid SettingWithCopyWarning
                condition = (df_out[col] < min_val) | (df_out[col] > max_val)
                df_out.loc[condition, col] = pd.NA
                new_nulls = df_out[col].isnull().sum()
                nan_count = new_nulls - original_nulls
                if nan_count > 0:
                    logger.warning(f"Found and nulled {nan_count} outliers in '{col}' outside the range {min_val}-{max_val}.")
        
        return df_out
