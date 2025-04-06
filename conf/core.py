from pathlib import Path
from typing import List, Tuple

from pydantic import BaseModel
from strictyaml import YAML, load

# Project directories

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
RECO_DIR = PACKAGE_ROOT / "recommendations" 
DATA_DIR = PACKAGE_ROOT / "data"
SAVED_MODEL_DIR = PACKAGE_ROOT / "saved_models"
TRAINING_EXP_DIR = PACKAGE_ROOT / "experiments"
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"

EXP2PARTNERS_TABLE_PATH = DATA_DIR / "experiences_and_partners.csv"
TAGGED_PARTNERS_TABLE_PATH = DATA_DIR / "tagged_partners.csv"
USER_ITEM_DATASET_PATH = DATA_DIR / "user_item_dataset.pkl"

class ParamGrid(BaseModel):
    """Parameters to supply for grid search in parameter optimization"""
    no_components: List[int]
    loss : Tuple[str, str]
    item_alpha: List[float]
    learning_schedule: Tuple[str, str]
    random_state: List[int]
    epochs: List[int]

class ModelConfig(BaseModel):
    """All configuration parameters relevant to model training and feature engineering"""
    no_components: int
    loss : str
    item_alpha: float
    learning_schedule: str
    random_state: int
    epochs: int
    model_file_name: str = None

class ExperimentConfig(BaseModel):
    """Configuration for each training experiment"""

    experiment_name: str
    model_type: str= 'hybrid'
    model_config: ModelConfig
    param_opt: bool = True
    param_grid: ParamGrid
    test_size: float


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None):
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    _config = ExperimentConfig(**parsed_config.data)

    return _config


config = create_and_validate_config()