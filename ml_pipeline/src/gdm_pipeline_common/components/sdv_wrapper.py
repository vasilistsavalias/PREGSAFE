from sdv.sampling import Condition
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from gdm_pipeline_common.config_manager import ConfigManager
import pandas as pd

class CTGANWrapper:
    def __init__(self, params: dict):
        self.params = params
        self.metadata = SingleTableMetadata()
        self.model = None
        self.preprocessor = None

    def fit(self, data: pd.DataFrame):
        self.metadata.detect_from_dataframe(data=data)
        self.model = CTGANSynthesizer(self.metadata, **self.params)
        self.model.fit(data)

    def sample(self, num_rows: int = None, conditions: list = None) -> pd.DataFrame:
        if conditions:
            # The sample_from_conditions method in sdv doesn't use batch_size, etc.
            # It's simpler than the general sample method.
            return self.model.sample_from_conditions(conditions=conditions)
        elif num_rows:
            return self.model.sample(num_rows=num_rows)
        else:
            raise ValueError("Either num_rows or conditions must be provided.")

    def save(self, filepath: str):
        self.model.save(filepath)


class TVAEWrapper:
    def __init__(self, params: dict):
        self.params = params
        self.metadata = SingleTableMetadata()
        self.model = None
        self.preprocessor = None

    def fit(self, data: pd.DataFrame):
        self.metadata.detect_from_dataframe(data=data)
        self.model = TVAESynthesizer(self.metadata, **self.params)
        self.model.fit(data)

    def sample(self, num_rows: int = None, conditions: list = None) -> pd.DataFrame:
        if conditions:
            return self.model.sample_from_conditions(conditions=conditions)
        elif num_rows:
            return self.model.sample(num_rows=num_rows)
        else:
            raise ValueError("Either num_rows or conditions must be provided.")
