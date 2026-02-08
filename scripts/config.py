"""
Configuration file for project paths.
"""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Subdirectory paths
DATA_DIR = PROJECT_ROOT / "data"
REQUESTS_DIR = DATA_DIR / "requests"
RESPONSES_DIR = DATA_DIR / "responses"
DOCUMENTATION_DIR = PROJECT_ROOT / "documentation"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
