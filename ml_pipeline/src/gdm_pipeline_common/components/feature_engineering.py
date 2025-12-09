import pandas as pd
from loguru import logger
import numpy as np

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = logger

    def run(self) -> pd.DataFrame:
        """
        Runs the full feature engineering pipeline.
        """
        self.logger.info("--- Starting Feature Engineering ---")
        self._engineer_bmi_features()
        self._engineer_parity_feature()
        self._engineer_maternal_age_feature()
        self._engineer_weight_gain_features()
        self._engineer_risk_factor_count()
        self.logger.info("--- Feature Engineering Finished ---")
        return self.df

    def _engineer_bmi_features(self):
        """Engineers BMI and related categorical features."""
        self.logger.info("Engineering BMI features...")
        if 'Wt pre' in self.df.columns and 'Ht' in self.df.columns:
            # Ensure Ht is not zero to avoid division errors
            self.df['Ht'] = self.df['Ht'].replace(0, np.nan)
            self.df['BMI'] = self.df['Wt pre'] / ((self.df['Ht'] / 100) ** 2)
            self.logger.info("  - Engineered 'BMI' feature.")
            
            # This logic was present in the old notebooks for creating BMI categories
            bins = [0, 18.5, 25, 30, 150]
            labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
            self.df['BMI_Category'] = pd.cut(self.df['BMI'], bins=bins, labels=labels, right=False)
            self.logger.info("  - Engineered 'BMI_Category' feature.")

    def _engineer_parity_feature(self):
        """Engineers binary parity feature."""
        self.logger.info("Engineering Parity feature...")
        if 'Parity' in self.df.columns:
            self.df['Parity01'] = (self.df['Parity'] > 0).astype(int)
            self.logger.info("  - Engineered 'Parity01'.")

    def _engineer_maternal_age_feature(self):
        """Engineers binary and squared maternal age features."""
        self.logger.info("Engineering Maternal Age features...")
        if 'MA' in self.df.columns:
            self.df['MA>35 01'] = (self.df['MA'] > 35).astype(int)
            self.df['MA_squared'] = self.df['MA'] ** 2
            self.logger.info("  - Engineered 'MA>35 01' and 'MA_squared'.")

    def _engineer_weight_gain_features(self):
        """Engineers rate and relative weight gain features."""
        self.logger.info("Engineering Weight Gain features...")
        if 'Wgain' in self.df.columns and 'GA days' in self.df.columns:
            self.df['GA days'] = self.df['GA days'].replace(0, np.nan)
            self.df['Weight_Gain_Rate'] = self.df['Wgain'] / (self.df['GA days'] / 7)
            self.logger.info("  - Engineered 'Weight_Gain_Rate'.")
        
        if 'Wgain' in self.df.columns and 'Wt pre' in self.df.columns:
            self.df['Wt pre'] = self.df['Wt pre'].replace(0, np.nan)
            self.df['Relative_Wgain'] = self.df['Wgain'] / self.df['Wt pre']
            self.logger.info("  - Engineered 'Relative_Wgain'.")

    def _engineer_risk_factor_count(self):
        """Engineers the composite risk factor count."""
        self.logger.info("Engineering Risk Factor Count...")
        risk_factors = ['Smoking01', 'Conception ART 01', 'Thyroid all']
        existing_factors = [f for f in risk_factors if f in self.df.columns]
        if existing_factors:
            self.df['Risk_Factor_Count'] = self.df[existing_factors].sum(axis=1)
            self.logger.info(f"  - Engineered 'Risk_Factor_Count' from {len(existing_factors)} features.")