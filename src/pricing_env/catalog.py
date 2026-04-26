"""Catalog loading and configuration sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
import random

import yaml

from .types import CatalogOption


@dataclass
class Catalog:
    """Holds canonical customization options and sampling helpers.

    Attributes:
        options: All canonical options.
    """

    options: List[CatalogOption]
    implementation_cost_ratio: float = 0.5

    def by_dimension(self) -> Dict[str, List[CatalogOption]]:
        """Groups options by dimension.

        Returns:
            Mapping from dimension name to available options.
        """

        grouped: Dict[str, List[CatalogOption]] = {}
        for option in self.options:
            grouped.setdefault(option.dimension, []).append(option)
        return grouped

    def sample_configuration(self, rng: random.Random) -> List[CatalogOption]:
        """Samples one option per dimension for an episode.

        Args:
            rng: Random generator used for deterministic reproducibility.

        Returns:
            A list of selected canonical options.
        """

        selected: List[CatalogOption] = []
        for candidates in self.by_dimension().values():
            selected.append(rng.choice(candidates))
        return selected

    @staticmethod
    def total_msrp_delta(options: Iterable[CatalogOption]) -> float:
        """Computes total MSRP deltas for selected options.

        Args:
            options: Selected options.

        Returns:
            Sum of option MSRP deltas.
        """

        return float(sum(o.price_delta_usd for o in options))

    def total_implementation_cost(self, options: Iterable[CatalogOption]) -> float:
        """Computes seller-side implementation-cost proxy for selected options.

        Args:
            options: Selected options.

        Returns:
            Sum of implementation-cost proxies (`msrp_delta * ratio`).
        """

        return float(self.total_msrp_delta(options) * float(self.implementation_cost_ratio))

    def total_cost(self, options: Iterable[CatalogOption]) -> float:
        """Computes total implementation cost (compatibility alias).

        Args:
            options: Selected options.

        Returns:
            Sum of implementation-cost proxies.
        """

        return self.total_implementation_cost(options)

    @staticmethod
    def aesthetic_proxy(options: Iterable[CatalogOption]) -> float:
        """Computes mean aesthetics proxy score.

        Args:
            options: Selected options.

        Returns:
            Mean of aesthetics weights or 0.0 if empty.
        """

        options_list = list(options)
        if not options_list:
            return 0.0
        return float(sum(o.aesthetic_weight for o in options_list) / len(options_list))


def load_catalog(path: str | Path) -> Catalog:
    """Loads canonical customization options from YAML.

    Args:
        path: YAML file path.

    Returns:
        Parsed catalog object.
    """

    data = yaml.safe_load(Path(path).read_text())
    options = [
        CatalogOption(
            key=item["key"],
            dimension=item["dimension"],
            price_delta_usd=float(item["price_delta_usd"]),
            aesthetic_weight=float(item["aesthetic_weight"]),
        )
        for item in data["options"]
    ]
    return Catalog(options=options)
