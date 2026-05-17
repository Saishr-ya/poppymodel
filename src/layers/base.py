"""
src/layers/base.py

Abstract base class for all scoring layers.
Every layer must:
  1. Inherit from BaseLayer
  2. Set layer_name and version
  3. Implement score(pair) → CandidatePair

Design principles:
- Layers never crash the pipeline. Failures are logged and the pair is returned unchanged.
- Disqualified pairs skip expensive computation.
- Every layer records its data source and version in pair.data_sources.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import datetime
import logging

from src.scoring.candidate import CandidatePair


class BaseLayer(ABC):
    """
    Base class for all scoring layers.

    Usage:
        class MyLayer(BaseLayer):
            layer_name = "my_layer"
            version = "1.0"

            def score(self, pair: CandidatePair) -> CandidatePair:
                # compute, mutate pair.scores / pair.flags
                return pair

        layer = MyLayer(config={})
        pair = layer.run(pair)
    """

    layer_name: str = "base"
    version: str = "1.0"

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, pair: CandidatePair) -> CandidatePair:
        """
        Public entry point. Handles skip logic and error isolation.
        Do not override — override score() instead.
        """
        # Skip disqualified pairs (saves expensive API calls)
        if pair.flags.is_disqualified:
            self.logger.debug(
                f"[{self.layer_name}] Skipping disqualified pair "
                f"{pair.drug_id}×{pair.disease_id}: {pair.flags.disqualify_reason}"
            )
            return pair

        try:
            pair = self.score(pair)
            pair.data_sources[self.layer_name] = {
                "version": self.version,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "status": "ok",
            }
            pair.touch()
        except Exception as exc:
            self.logger.error(
                f"[{self.layer_name}] FAILED for {pair.drug_id}×{pair.disease_id}: {exc}",
                exc_info=True,
            )
            pair.data_sources[self.layer_name] = {
                "version": self.version,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "status": "error",
                "error": str(exc),
            }

        return pair

    @abstractmethod
    def score(self, pair: CandidatePair) -> CandidatePair:
        """
        Implement this method in each layer subclass.
        Mutate pair.scores and/or pair.flags.
        Return the mutated pair.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(v{self.version})"
