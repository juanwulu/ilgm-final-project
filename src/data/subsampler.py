# Copyright (c) 2024, Juanwu Lu.
# Released under the BSD 3-Clause License.
# Please see the LICENSE file that should have been included as part of this
# project source code.
"""INTERACTION dataset scenario filter module."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

INTERACTION_MAP_NAME_MAPPING: Dict[str, List[str]] = {
    "train": [
        "DR_CHN_Merging_ZS0",
        "DR_CHN_Merging_ZS2",
        "DR_CHN_Roundabout_LN",
        "DR_DEU_Merging_MT",
        "DR_DEU_Roundabout_OF",
        "DR_USA_Intersection_EP0",
        "DR_USA_Intersection_EP1",
        "DR_USA_Intersection_GL",
        "DR_USA_Intersection_MA",
        "DR_USA_Roundabout_EP",
        "DR_USA_Roundabout_FT",
        "DR_USA_Roundabout_SR",
    ],
    "val": [
        "DR_CHN_Merging_ZS0",
        "DR_CHN_Merging_ZS2",
        "DR_CHN_Roundabout_LN",
        "DR_DEU_Merging_MT",
        "DR_DEU_Roundabout_OF",
        "DR_USA_Intersection_EP0",
        "DR_USA_Intersection_EP1",
        "DR_USA_Intersection_GL",
        "DR_USA_Intersection_MA",
        "DR_USA_Roundabout_EP",
        "DR_USA_Roundabout_FT",
        "DR_USA_Roundabout_SR",
    ],
    "test": [
        "DR_CHN_Merging_ZS0",
        "DR_CHN_Merging_ZS2",
        "DR_CHN_Roundabout_LN",
        "DR_DEU_Merging_MT",
        "DR_DEU_Roundabout_OF",
        "DR_USA_Intersection_EP0",
        "DR_USA_Intersection_EP1",
        "DR_USA_Intersection_GL",
        "DR_USA_Intersection_MA",
        "DR_USA_Roundabout_EP",
        "DR_USA_Roundabout_FT",
        "DR_USA_Roundabout_SR",
        "DR_Intersection_CM",
        "DR_LaneChange_ET0",
        "DR_LaneChange_ET1",
        "DR_Merging_TR0",
        "DR_Merging_TR1",
        "DR_Roundabout_RW",
    ],
    "all": [
        "DR_CHN_Merging_ZS0",
        "DR_CHN_Merging_ZS2",
        "DR_CHN_Roundabout_LN",
        "DR_DEU_Merging_MT",
        "DR_DEU_Roundabout_OF",
        "DR_USA_Intersection_EP0",
        "DR_USA_Intersection_EP1",
        "DR_USA_Intersection_GL",
        "DR_USA_Intersection_MA",
        "DR_USA_Roundabout_EP",
        "DR_USA_Roundabout_FT",
        "DR_USA_Roundabout_SR",
        "DR_Intersection_CM",
        "DR_LaneChange_ET0",
        "DR_LaneChange_ET1",
        "DR_Merging_TR0",
        "DR_Merging_TR1",
        "DR_Roundabout_RW",
    ],
}


@dataclass(frozen=True)
class INTERACTIONSubSampler:
    """A dataclass storing configs for filtering INTERACTION dataset scenarios.

    Attributes:
        ratio (float): a float subsample ratio between 0 and 1.
        locations (Optional[List[str]], optional): an optional list of location
            names to include. If it is `None`, use all available locations.
            Defaults to `None`.
    """

    ratio: float = field(default=1.0)
    locations: Optional[List[str]] = field(default=None)

    def __post_init__(self) -> None:
        assert (
            isinstance(self.ratio, float)
            and self.ratio > 0
            and self.ratio <= 1
        ), ValueError(f"Invalid subsample ratio {self.ratio}")
