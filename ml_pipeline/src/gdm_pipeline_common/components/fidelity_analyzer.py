import pandas as pd
from loguru import logger
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split

class FidelityAnalyzer:
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, eval_config: dict, synthesizer_name: str, output_dir: Path, logger):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.eval_config = eval_config
        self.synthesizer_name = synthesizer_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def run_all_analyses(self):
        """Runs all configured fidelity analyses."""
        self.logger.info(f"--- Starting Fidelity Analysis for {self.synthesizer_name} ---")
        if self.eval_config.get('generate_pca_plot', False):
            self.plot_pca()
        if self.eval_config.get('generate_propensity_plot', False):
            self.calculate_and_plot_propensity_score()
        if self.eval_config.get('generate_correlation_heatmaps', False):
            self.plot_correlation_heatmaps()
        if self.eval_config.get('generate_feature_distribution_plots', False):
            self.plot_feature_distributions()
        self.logger.info(f"--- Fidelity Analysis for {self.synthesizer_name} Finished ---")

    def plot_pca(self):
        """Generates and saves a PCA plot comparing real and synthetic data."""
        self.logger.info("Generating PCA plot...")
        try:
            pca = PCA(n_components=2)
            
            real_pca = pca.fit_transform(self.real_data)
            synth_pca = pca.transform(self.synthetic_data)
            
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=real_pca[:, 0], y=real_pca[:, 1], alpha=0.7, label='Real Data')
            sns.scatterplot(x=synth_pca[:, 0], y=synth_pca[:, 1], alpha=0.7, label='Synthetic Data')
            plt.title(f'PCA Comparison: Real vs. Synthetic ({self.synthesizer_name})')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend()
            save_path = self.output_dir / "pca_comparison.png"
            plt.savefig(save_path)
            plt.close()
            self.logger.info(f"Saved PCA plot to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate PCA plot: {e}")

    def calculate_and_plot_propensity_score(self):
        """Trains a classifier to distinguish real vs. synthetic data."""
        self.logger.info("Calculating Propensity Score (pMSE)...")
        try:
            real = self.real_data.copy()
            synth = self.synthetic_data.copy()
            real['is_real'] = 1
            synth['is_real'] = 0
            
            combined_df = pd.concat([real, synth], ignore_index=True)
            X = combined_df.drop(columns=['is_real'])
            y = combined_df['is_real']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
            pmse = mean_squared_error(y_test, probs)
            
            self.logger.info(f"Propensity Score AUC: {auc:.4f} (Closer to 0.5 is better)")
            self.logger.info(f"Propensity Score pMSE: {pmse:.4f} (Closer to 0.25 is better)")

            # Plotting logic can be added here if desired
            
        except Exception as e:
            self.logger.error(f"Failed to calculate propensity score: {e}")

    def plot_correlation_heatmaps(self):
        """Generates and saves correlation heatmaps for real, synthetic, and difference."""
        self.logger.info("Generating correlation heatmaps...")
        try:
            # Real Data
            corr_real = self.real_data.corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_real, cmap='coolwarm', annot=False)
            plt.title(f'Correlation Matrix - Real Data ({self.synthesizer_name})')
            save_path_real = self.output_dir / "corr_real.png"
            plt.savefig(save_path_real)
            plt.close()

            # Synthetic Data
            corr_synth = self.synthetic_data.corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_synth, cmap='coolwarm', annot=False)
            plt.title(f'Correlation Matrix - Synthetic Data ({self.synthesizer_name})')
            save_path_synth = self.output_dir / "corr_synthetic.png"
            plt.savefig(save_path_synth)
            plt.close()

            # Difference
            corr_diff = (corr_real - corr_synth).abs()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_diff, cmap='viridis', annot=False)
            plt.title(f'Absolute Correlation Difference ({self.synthesizer_name})')
            save_path_diff = self.output_dir / "corr_difference.png"
            plt.savefig(save_path_diff)
            plt.close()
            
            self.logger.info(f"Saved correlation heatmaps to {self.output_dir}")
        except Exception as e:
            self.logger.error(f"Failed to generate correlation heatmaps: {e}")

    def plot_feature_distributions(self):
        """Generates and saves distribution plots for each feature."""
        self.logger.info("Generating feature distribution plots...")
        dist_plot_dir = self.output_dir / "feature_distributions"
        dist_plot_dir.mkdir(exist_ok=True)
        
        for col in self.real_data.columns:
            try:
                plt.figure(figsize=(10, 6))
                sns.kdeplot(self.real_data[col], label='Real', fill=True)
                sns.kdeplot(self.synthetic_data[col], label='Synthetic', fill=True)
                plt.title(f'Distribution Comparison for {col} ({self.synthesizer_name})')
                plt.legend()
                # Sanitize column name for filename
                clean_col = "".join(c for c in col if c.isalnum() or c in ('_', '-')).rstrip()
                save_path = dist_plot_dir / f"dist_{clean_col}.png"
                plt.savefig(save_path)
                plt.close()
            except Exception as e:
                self.logger.warning(f"Could not generate distribution plot for column '{col}': {e}")
        self.logger.info(f"Saved feature distribution plots to {dist_plot_dir}")
