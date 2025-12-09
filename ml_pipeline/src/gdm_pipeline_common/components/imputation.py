import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from loguru import logger
from gdm_pipeline_common.utils.common import save_pickle
from typing import List

class DataPreprocessor:
    def __init__(self, skewed_features: List[str], all_numeric_features: List[str], random_seed: int):
        """
        Initializes a preprocessor that handles skewed and non-skewed features differently.
        """
        self.skewed_features = skewed_features
        self.non_skewed_features = [f for f in all_numeric_features if f not in skewed_features]
        self.random_seed = random_seed
        self.preprocessor = self._build_preprocessor()
        logger.info("DataPreprocessor initialized.")

    def _build_preprocessor(self) -> ColumnTransformer:
        """Builds the ColumnTransformer with separate pipelines for skewed and non-skewed data."""
        logger.info("Building preprocessing pipelines...")
        
        skewed_pipe = Pipeline([
            ('transformer', PowerTransformer(method='yeo-johnson')),
            ('scaler', RobustScaler())
        ])
        logger.info(f"Skewed pipeline created for {len(self.skewed_features)} features.")

        nonskewed_pipe = Pipeline([
            ('scaler', RobustScaler())
        ])
        logger.info(f"Non-skewed pipeline created for {len(self.non_skewed_features)} features.")

        preprocessor = ColumnTransformer(
            transformers=[
                ('skewed', skewed_pipe, self.skewed_features),
                ('nonskewed', nonskewed_pipe, self.non_skewed_features)
            ],
            remainder='passthrough' # Keep any non-numeric columns (though there shouldn't be any)
        )
        logger.info("ColumnTransformer assembled.")
        return preprocessor

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the preprocessor on the data and transforms it."""
        logger.info("Fitting and transforming data with the preprocessor...")
        
        original_index = df.index
        processed_array = self.preprocessor.fit_transform(df)
        
        # Get feature names out to reconstruct the DataFrame correctly
        feature_names = self.preprocessor.get_feature_names_out()
        
        df_processed = pd.DataFrame(processed_array, columns=feature_names, index=original_index)
        
        logger.info("Preprocessing complete.")
        return df_processed

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data using the already fitted preprocessor."""
        logger.info("Transforming data with the fitted preprocessor...")
        
        original_index = df.index
        processed_array = self.preprocessor.transform(df)
        
        feature_names = self.preprocessor.get_feature_names_out()
        
        df_processed = pd.DataFrame(processed_array, columns=feature_names, index=original_index)
        
        logger.info("Transformation complete.")
        return df_processed

    def inverse_transform(self, df_processed: pd.DataFrame) -> pd.DataFrame:
        """Applies the inverse transform to return data to its original scale."""
        logger.info("Applying inverse transform to the data...")
        
        df_inversed = pd.DataFrame(index=df_processed.index)

        # Inverse transform skewed features
        skewed_cols_processed = [c for c in df_processed.columns if c.startswith('skewed__')]
        if skewed_cols_processed:
            skewed_data_processed = df_processed[skewed_cols_processed].to_numpy()
            skewed_pipe = self.preprocessor.named_transformers_['skewed']
            
            # Manually inverse transform step-by-step
            inversed_skewed_data = skewed_pipe.named_steps['scaler'].inverse_transform(skewed_data_processed)
            inversed_skewed_data = skewed_pipe.named_steps['transformer'].inverse_transform(inversed_skewed_data)
            
            df_inversed[self.skewed_features] = inversed_skewed_data

        # Inverse transform non-skewed features
        nonskewed_cols_processed = [c for c in df_processed.columns if c.startswith('nonskewed__')]
        if nonskewed_cols_processed:
            nonskewed_data_processed = df_processed[nonskewed_cols_processed].to_numpy()
            nonskewed_pipe = self.preprocessor.named_transformers_['nonskewed']
            
            # Manually inverse transform step-by-step
            inversed_nonskewed_data = nonskewed_pipe.named_steps['scaler'].inverse_transform(nonskewed_data_processed)

            df_inversed[self.non_skewed_features] = inversed_nonskewed_data

        # Handle remainder columns
        remainder_cols = [c for c in df_processed.columns if c.startswith('remainder__')]
        if remainder_cols:
            original_remainder_names = [c.replace('remainder__', '') for c in remainder_cols]
            df_inversed[original_remainder_names] = df_processed[remainder_cols].values

        logger.info("Inverse transform complete.")
        return df_inversed

    def save(self, path: str):
        """Saves the fitted preprocessor object."""
        save_pickle(self.preprocessor, path)
        logger.info(f"Preprocessor saved to {path}")
