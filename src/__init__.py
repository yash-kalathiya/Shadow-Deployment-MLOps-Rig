"""
Shadow MLOps Source Package

This package contains the core components for the shadow deployment
churn prediction system.
"""

from src.models import ChampionModel, ChallengerModel, ModelRegistry

__version__ = "1.0.0"
__all__ = ["ChampionModel", "ChallengerModel", "ModelRegistry"]
