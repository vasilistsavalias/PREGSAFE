
import pandas as pd
from pathlib import Path
from loguru import logger

from gdm_pipeline_common.utils.evaluation import pca_eigen_diff, convert_numpy_to_native
from gdm_pipeline_common.utils.common import save_yaml
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata

class QualityAnalyzer:
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, output_path: Path):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Runs the full suite of quantitative quality analyses."""
        logger.info("--- Running Quantitative Quality Analysis ---")
        
        all_results = {}

        # 1. SDMetrics Report
        logger.info("Generating SDMetrics Quality Report...")
        try:
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(self.real_data)
            report = QualityReport()
            report.generate(self.real_data, self.synthetic_data, metadata=metadata.to_dict())
            
            all_results["sdmetrics_report"] = {
                "overall_quality_score": report.get_score(),
                "properties": report.get_properties().to_dict(),
            }
            logger.info("SDMetrics report complete.")
        except Exception as e:
            logger.exception("SDMetrics analysis failed.")
            all_results["sdmetrics_report"] = {"error": str(e)}

        # 2. PCA Eigenspectrum Difference
        logger.info("Calculating PCA Eigenspectrum Difference...")
        try:
            all_results["pca_eigenspectrum_difference"] = pca_eigen_diff(self.real_data, self.synthetic_data)
            logger.info("PCA calculation complete.")
        except Exception as e:
            logger.exception("PCA analysis failed.")
            all_results["pca_eigenspectrum_difference"] = {"error": str(e)}

        # Save the final report
        final_results = convert_numpy_to_native(all_results)
        save_yaml(final_results, self.output_path)
        logger.success(f"Quantitative quality report saved to {self.output_path}")
