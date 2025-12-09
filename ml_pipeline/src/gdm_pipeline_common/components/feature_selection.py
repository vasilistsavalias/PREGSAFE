import pandas as pd
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # <-- ADD THIS LINE
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer

class FeatureSelector:
    def __init__(self, df: pd.DataFrame, target_column: str, config: dict):
        self.df = df.copy()
        self.target_column = target_column
        self.config = config
        self.logger = logger

    def run(self) -> pd.DataFrame:
        """
        Selects features based on the 'selected_features' list in the config.
        """
        self.logger.info("--- Starting Feature Selection ---")
        
        selected_features = self.config.get('selected_features')
        if not selected_features:
            self.logger.error("'selected_features' not found in the configuration. Cannot proceed.")
            raise ValueError("'selected_features' not found in config.")

        # Ensure the target column is included if it's not already in the list
        if self.target_column not in selected_features:
            self.logger.warning(f"Target column '{self.target_column}' not in selected_features list. Adding it.")
            # selected_features.append(self.target_column) # It should already be there

        # Select the columns from the dataframe
        self.logger.info(f"Selecting the following features as specified in config: {selected_features}")
        
        # Verify all selected features are present in the dataframe
        missing_features = [col for col in selected_features if col not in self.df.columns]
        if missing_features:
            self.logger.error(f"The following selected features are not in the dataframe: {missing_features}")
            raise ValueError(f"Missing features in dataframe: {missing_features}")
            
        df_final = self.df[selected_features]
        
        self.logger.info(f"Feature selection complete. Final shape: {df_final.shape}")
        self.logger.info("--- Feature Selection Finished ---")
        return df_final

    def _clean_for_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Performs imputation and one-hot encoding to prepare data for selection algorithms.
        """
        self.logger.info("Preparing data for selection (imputation and encoding)...")
        
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

        imputation_pipeline = ColumnTransformer(
            transformers=[
                ('num', IterativeImputer(max_iter=10, random_state=self.config.get('random_seed', 42)), numerical_features),
                ('cat', SimpleImputer(strategy='most_frequent'), categorical_features)
            ], remainder='passthrough'
        )
        X_imputed_array = imputation_pipeline.fit_transform(X)
        X_imputed = pd.DataFrame(X_imputed_array, columns=numerical_features + categorical_features, index=X.index)

        if categorical_features:
            encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
            encoded_cats_array = encoder.fit_transform(X_imputed[categorical_features])
            encoded_cat_names = encoder.get_feature_names_out(categorical_features)
            X_encoded_cats = pd.DataFrame(encoded_cats_array, columns=encoded_cat_names, index=X_imputed.index)
            X_cleaned = X_imputed.drop(columns=categorical_features).join(X_encoded_cats)
        else:
            X_cleaned = X_imputed
        
        self.logger.info(f"Data prepared. Shape for selection: {X_cleaned.shape}")
        return X_cleaned

    def _selection_gauntlet(self, X: pd.DataFrame, y: pd.Series) -> list:
        """
        Runs the three-stage feature selection process.
        """
        # Stage 1: Random Forest Importance
        self.logger.info("Gauntlet Stage 1: Screening with Random Forest Importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=self.config.get('random_seed', 42), n_jobs=-1)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        stage1_candidates = importances.nlargest(20).index.tolist()
        self.logger.info(f"Top 20 features from RF: {stage1_candidates}")

        # Stage 2: Multicollinearity Management
        self.logger.info("Gauntlet Stage 2: Managing Multicollinearity...")
        X_stage1 = X[stage1_candidates]
        corr_matrix = X_stage1.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
        stage2_candidates = [f for f in stage1_candidates if f not in to_drop]
        self.logger.info(f"{len(stage1_candidates) - len(stage2_candidates)} features dropped. {len(stage2_candidates)} remain.")

        # Stage 3: Recursive Feature Elimination (RFE)
        self.logger.info("Gauntlet Stage 3: Final Selection with RFE...")
        if len(stage2_candidates) < 12:
            self.logger.warning("Fewer than 12 candidates remain, skipping RFE.")
            final_features = stage2_candidates
        else:
            estimator = RandomForestClassifier(n_estimators=50, random_state=self.config.get('random_seed', 42), n_jobs=-1)
            rfe = RFE(estimator, n_features_to_select=12, step=1)
            rfe.fit(X[stage2_candidates], y)
            final_features = list(pd.Series(stage2_candidates)[rfe.support_])
        
        self.logger.info(f"Final selected features: {final_features}")
        return final_features
