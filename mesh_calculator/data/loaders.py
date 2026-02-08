"""
Data loading utilities.
"""
import yaml
from typing import Dict, Any

from ..core.config import MeshCalculatorConfig, InputPaths, OutputPaths, MeshConfig


def load_config(config_path: str) -> MeshCalculatorConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        MeshCalculatorConfig object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return MeshCalculatorConfig.from_dict(config_dict)
