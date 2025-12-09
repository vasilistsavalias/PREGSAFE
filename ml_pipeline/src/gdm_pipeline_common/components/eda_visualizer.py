import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np

class EDAVisualizer:
    def __init__(self, df: pd.DataFrame, config: dict, output_dir: Path):
        """
        Initializes the EDAVisualizer.

        Args:
            df (pd.DataFrame): The raw dataframe to analyze.
            config (dict): The project configuration dictionary.
            output_dir (Path): The directory to save the plots.
        """
        self.df = df.copy()
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_col = self.config['global_params']['target_column']
        
        # Define key features for focused analysis
        self.key_features = [
            'MA', 'Parity', 'Wt pre', 'Ht', 'Wgain', 'BMI', 
            'Weight_Gain_Rate', 'Relative_Wgain', 'Risk_Factor_Count'
        ]
        logger.info("EDAVisualizer initialized.")

    def _perform_basic_feature_engineering(self):
        """Performs basic feature engineering required for EDA context."""
        logger.info("Performing basic feature engineering for EDA context...")
        # Use .loc to avoid SettingWithCopyWarning
        self.df.loc[:, 'BMI'] = self.df['Wt pre'] / ((self.df['Ht'] / 100) ** 2)
        self.df.loc[:, 'Weight_Gain_Rate'] = self.df['Wgain'] / (self.df['GA days'] / 7) if 'GA days' in self.df.columns else 0
        self.df.loc[:, 'Relative_Wgain'] = (self.df['Wgain'] / self.df['Wt pre']) * 100 if 'Wt pre' in self.df.columns else 0
        risk_factors = [col for col in ['Thyroid all', 'Conception ART 01', 'Smoking01'] if col in self.df.columns]
        self.df.loc[:, 'Risk_Factor_Count'] = self.df[risk_factors].sum(axis=1)
        logger.info("Basic feature engineering complete.")

    def run_visualizations(self):
        """
        Generates and saves all the key visualizations for the EDA.
        """
        logger.info("--- Starting EDA Visualization Generation ---")
        self._perform_basic_feature_engineering()

        self._plot_target_distribution()
        self._plot_univariate_distributions()
        self._plot_bivariate_analysis()
        self._plot_correlation_heatmap()
        self._plot_pairplot()

        logger.info(f"--- EDA Visualizations saved in {self.output_dir} ---")

    def _plot_target_distribution(self):
        """Plots the distribution of the target variable."""
        logger.info("Plotting target variable distribution...")
        plt.figure(figsize=(8, 6))
        sns.countplot(x=self.target_col, data=self.df)
        plt.title('Target Variable (GDM01) Distribution')
        plt.savefig(self.output_dir / "target_distribution.png")
        plt.close()

    def _plot_univariate_distributions(self):
        """Plots histograms for key numerical features."""
        logger.info("Plotting univariate distributions...")
        univariate_dir = self.output_dir / "univariate_plots"
        univariate_dir.mkdir(exist_ok=True)
        for feature in self.key_features:
            if feature in self.df.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(self.df[feature], kde=True)
                plt.title(f'Distribution of {feature}')
                plt.savefig(univariate_dir / f"{feature}_distribution.png")
                plt.close()

    def _plot_bivariate_analysis(self):
        """Plots boxplots of key features against the target variable."""
        logger.info("Plotting bivariate analysis (feature vs. target)...")
        bivariate_dir = self.output_dir / "bivariate_plots"
        bivariate_dir.mkdir(exist_ok=True)
        for feature in self.key_features:
            if feature in self.df.columns:
                plt.figure(figsize=(12, 7))
                sns.boxplot(x=self.target_col, y=feature, data=self.df)
                plt.title(f'{feature} vs. {self.target_col}')
                plt.savefig(bivariate_dir / f"{feature}_vs_target.png")
                plt.close()

    def _plot_correlation_heatmap(self):
        """Plots a correlation heatmap of key numerical features."""
        logger.info("Plotting correlation heatmap...")
        plt.figure(figsize=(16, 12))
        # Ensure only existing key features are used
        existing_key_features = [f for f in self.key_features if f in self.df.columns]
        corr = self.df[existing_key_features].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix of Key Features')
        plt.savefig(self.output_dir / "correlation_heatmap.png")
        plt.close()

    def _plot_pairplot(self):
        """
        Plots a pairplot of the top N most correlated features with the target.
        """
        logger.info("Generating pairplot for top correlated features...")
        
        # Get the list of final predictors from the config
        final_predictors = self.config.get('feature_params', {}).get('final_predictor_list', [])
        
        # Ensure predictors exist in the dataframe and are numeric
        numeric_predictors = [f for f in final_predictors if f in self.df.columns and pd.api.types.is_numeric_dtype(self.df[f])]
        
        if not numeric_predictors:
            logger.warning("No numeric predictors found to generate a pairplot.")
            return
            
        # Calculate correlation with the target variable
        correlations = self.df[numeric_predictors].corrwith(self.df[self.target_col]).abs().sort_values(ascending=False)
        
        # Select the top 5 features (or fewer if not available)
        top_n = min(5, len(correlations))
        top_features = correlations.head(top_n).index.tolist()
        
        # Add the target column for hue
        plot_features = top_features + [self.target_col]
        
        logger.info(f"Selected top {top_n} features for pairplot: {top_features}")

        if len(top_features) > 1:
            try:
                sns.pairplot(self.df[plot_features], hue=self.target_col, corner=True)
                plt.suptitle('Pairplot of Top Correlated Features', y=1.02)
                plt.savefig(self.output_dir / "pairplot.png")
            except Exception as e:
                logger.error(f"Failed to generate pairplot: {e}")
            finally:
                plt.close()
        else:
            logger.warning("Not enough correlated features available to generate a pairplot.")

if __name__ == '__main__':
    # This is for standalone execution/testing
    from gdm_pipeline.config_manager import ConfigManager
    from gdm_pipeline.utils.common import load_dataset

    try:
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        raw_data_path = Path(config['data_ingestion']['source_path'])
        df_raw = load_dataset(raw_data_path)
        
        eda_output_dir = Path(config['artifacts_root']) / "00_eda"
        
        visualizer = EDAVisualizer(df=df_raw, config=config, output_dir=eda_output_dir)
        visualizer.run_visualizations()
        
    except Exception as e:
        logger.exception("Failed to run standalone EDA script.")
