from gdm_pipeline_common.utils.common import load_yaml
from pathlib import Path
from box import ConfigBox
import copy # Import the copy module for deepcopy

class ConfigManager:
    def __init__(self, config_filepath: Path = Path("config/config.yaml"), smoke_test_override: bool = False):
        """
        Initializes the ConfigManager and prepares the final configuration.
        It intelligently merges smoke_test or full_run parameters into the main
        config structure based on the smoke_test_override flag.
        """
        base_config = load_yaml(config_filepath)
        self.smoke_test_override = smoke_test_override
        # The root of the processing now starts with a deepcopy.
        self.config = self._process_config(copy.deepcopy(base_config))

    def _process_config(self, config_node: ConfigBox) -> ConfigBox:
        """
        Recursively processes the config to merge run-specific parameters.
        This function now operates safely on deep copies.
        """
        run_mode = "smoke_test" if self.smoke_test_override else "full_run"

        # Recurse first on all nested dictionaries
        for key, value in config_node.items():
            if isinstance(value, ConfigBox):
                config_node[key] = self._process_config(value)

        # After recursion, check if the current node has run-specific keys to merge
        if run_mode in config_node:
            config_node.merge_update(config_node[run_mode])
            config_node.pop("smoke_test", None)
            config_node.pop("full_run", None)
            
        return config_node

    def get_config(self) -> ConfigBox:
        """Returns the entire, processed configuration dictionary."""
        return self.config
