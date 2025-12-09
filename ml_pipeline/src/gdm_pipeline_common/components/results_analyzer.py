import pandas as pd
from loguru import logger
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon

class ResultsAnalyzer:
    def __init__(self, df: pd.DataFrame, config: dict, eval_config: dict):
        self.df = df
        self.config = config
        self.eval_config = eval_config
        self.primary_metrics = self.eval_config.get('primary_metrics', ['roc_auc'])
        self.output_dir = Path(self.config['analysis_plots_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def run_analysis(self):
        """Runs the full analysis and generates all plots."""
        self.logger.info("--- Starting Results Analysis and Visualization ---")
        for metric in self.primary_metrics:
            self._plot_performance_stability(metric)
            self._plot_mean_performance_comparison(metric)
        
        self._perform_statistical_test()
        self.logger.info("--- Analysis and Visualization Finished ---")

    def _plot_performance_stability(self, metric: str):
        """Creates boxplots to show the stability of models across folds for a given metric."""
        self.logger.info(f"Generating performance stability boxplots for metric: {metric}...")
        try:
            plt.figure(figsize=(15, 8))
            sns.boxplot(data=self.df, x='model', y=metric, hue='method')
            plt.title(f'Model Performance Stability ({metric}) Across Folds')
            plt.xticks(rotation=45)
            plt.tight_layout()
            save_path = self.output_dir / f"performance_stability_{metric.lower()}.png"
            plt.savefig(save_path)
            plt.close()
            self.logger.info(f"Saved stability plot to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate stability plot for {metric}: {e}")

    def _plot_mean_performance_comparison(self, metric: str):
        """Creates bar plots to compare the mean performance for a given metric."""
        self.logger.info(f"Generating mean performance comparison bar plots for metric: {metric}...")
        try:
            mean_scores = self.df.groupby(['method', 'model'])[metric].mean().reset_index()
            
            plt.figure(figsize=(15, 8))
            sns.barplot(data=mean_scores, x='model', y=metric, hue='method')
            plt.title(f'Mean Model Performance Comparison ({metric})')
            plt.xticks(rotation=45)
            plt.ylabel(f'Mean {metric}')
            plt.tight_layout()
            save_path = self.output_dir / f"mean_performance_comparison_{metric.lower()}.png"
            plt.savefig(save_path)
            plt.close()
            self.logger.info(f"Saved mean performance plot to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate mean performance plot for {metric}: {e}")

    def _perform_statistical_test(self):
        """Performs a Wilcoxon signed-rank test comparing TRTR vs. the best TSTR method."""
        self.logger.info("Performing statistical significance testing...")
        try:
            # Determine the best TSTR method overall based on the first primary metric
            primary_metric = self.primary_metrics[0]
            mean_scores = self.df.groupby('method')[primary_metric].mean()
            tstr_methods = [m for m in mean_scores.index if 'TSTR' in m]
            if not tstr_methods:
                self.logger.warning("No TSTR methods found for statistical test.")
                return
            
            best_tstr_method = mean_scores.loc[tstr_methods].idxmax()
            self.logger.info(f"Best TSTR method for comparison: {best_tstr_method}")

            # Get the scores for the baseline (TRTR) and the best TSTR method
            # Assuming the same champion model is used for both for a fair comparison
            champion_model = self.df[self.df['method'] == best_tstr_method]['model'].iloc[0]
            
            trtr_scores = self.df[(self.df['method'] == 'TRTR') & (self.df['model'] == champion_model)][primary_metric]
            tstr_scores = self.df[(self.df['method'] == best_tstr_method) & (self.df['model'] == champion_model)][primary_metric]

            if len(trtr_scores) != len(tstr_scores) or len(trtr_scores) < 2:
                self.logger.warning("Not enough paired data to perform statistical test.")
                return

            # Perform the Wilcoxon signed-rank test
            stat, p_value = wilcoxon(tstr_scores, trtr_scores, alternative='greater')
            
            # Create and save the summary plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.6, f'P-Value: {p_value:.4f}', ha='center', va='center', fontsize=24, weight='bold')
            ax.text(0.5, 0.45, f'Test: Wilcoxon signed-rank (paired)', ha='center', va='center', fontsize=14)
            ax.text(0.5, 0.35, f'Comparison: TRTR vs {best_tstr_method} ({primary_metric})', ha='center', va='center', fontsize=14)
            
            result_text = f'Result is STATISTICALLY SIGNIFICANT at α=0.05' if p_value < 0.05 else 'Result is NOT statistically significant at α=0.05'
            bbox_props = dict(boxstyle="round,pad=0.3", fc="lightgreen" if p_value < 0.05 else "lightcoral", ec="black", lw=1)
            ax.text(0.5, 0.2, result_text, ha='center', va='center', fontsize=14, bbox=bbox_props)
            
            ax.set_title('Statistical Significance Test Results', fontsize=18)
            ax.axis('off')
            plt.tight_layout()
            save_path = self.output_dir / "statistical_test_summary.png"
            plt.savefig(save_path)
            plt.close()
            self.logger.info(f"Saved statistical test summary plot to {save_path}")

        except Exception as e:
            self.logger.error(f"Failed to perform statistical test: {e}")
