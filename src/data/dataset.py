# Copyright (c) 2024, Juanwu Lu.
# Released under the BSD 3-Clause License.
# Please see the LICENSE file that should have been included as part of this
# project source code.
"""UC Berkeley INTERACTION Dataset."""
from __future__ import annotations

import gzip
import pickle
import random
import traceback
from collections.abc import Iterable
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from interaction.dataset import (
    SPLITS,
    INTERACTIONCase,
    INTERACTIONMap,
    INTERACTIONScenario,
)
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from torch import Tensor
from torch_geometric.data import Dataset
from torch_geometric.transforms import BaseTransform, Compose

from src.utils.logging import get_pylogger

from .base_data import BaseData
from .subsampler import INTERACTIONSubSampler

# Constants
LOGGER = get_pylogger(__name__)
WAY_TYPE_OF_INTEREST = (
    "CURBSTONE_LOW",
    "GUARD_RAIL",
    "ROAD_BORDER",
    "LINE_THIN_SOLID",
    "LINE_THIN_SOLID_SOLID",
    "LINE_THIN_DASHED",
    "LINE_THICK_SOLID",
    "LINE_THICK_SOLID_SOLID",
    "STOP_LINE",
    "PEDESTRIAN_MARKING",
    "VIRTUAL",
    "VIRTUAL_SOLID",
)


class INTERACTIONData(BaseData):
    """Static traffic graph data for the INTERACTION dataset.

    .. note::
        This class extends the :class:`BaseStaticData` class by implementing
        multiple abstract properties:

        * map_feature_dims: in INTERACTION dataset, the number is `15`
        * motion_feature_dims: in INTERACTION dataset, the number is `2`
        * target_dims: in INTERACTION dataset, the number is `2`
    """

    @property
    def map_feature_dims(self) -> int:
        return 11

    @property
    def motion_feature_dims(self) -> int:
        return 8

    @property
    def target_dims(self) -> int:
        return 2


class INTERACTIONDataset(Dataset):
    """Dataset class of the INTERACTION dataset."""

    # ----------- public attributes ------------ #
    challenge_type: str
    """str: name of the challenge."""
    split: Literal["train", "val", "test"]
    """Literal["train", "val", "test"]: tag of the dataset."""
    subsampler: INTERACTIONSubSampler
    """INTERACTIONSubSampler: subsampler for the dataset."""
    radius: float
    """float: query range in meters."""
    force_data_cache: bool
    """bool: If save cache data to local."""
    train_on_multi_agent: bool
    """bool: if train on multi-agent prediction."""

    # ----------- private attributes ------------ #
    _indexer: List[str, int, int]
    """List[str, int, int]: metainfo indexer of the dataset."""
    _map_api_container: Dict[str, INTERACTIONMap]
    """Dict[str, INTERACTIONMap]: map api container {location:, api}."""
    _track_api_container: Dict[str, INTERACTIONScenario]
    """Dict[str, INTERACTIONScenario]: track api container {location: api}"""

    def __init__(
        self,
        root: str,
        challenge_type: str,
        subsampler: INTERACTIONSubSampler,
        split: str = "train",
        transform: Optional[BaseTransform | Iterable[BaseTransform]] = None,
        radius: float = 50,
        force_data_cache: bool = False,
        train_on_multi_agent: bool = False,
    ) -> None:
        """Constructor function.

        Args:
            root (str): root directory of the INTERACTION dataset files.
            challenge_type (str): name of the challenge, either `single-agent`,
            `conditional-single-agent`, `multi-agent`, or
            `conditional-multi-agent`.
            tag (str): tag of the dataset, either `train`, `val`, or `test`.
            subsampler (INTERACTIONSubSampler): subsampler for the dataset.
            radius (float, optional): query range in meters. Defaults to 50.
            force_data_cache (bool, optional): if save cache tensors to local.
            Defaults to False.
            num_workers (int): number of workers for multiprocessing.
            Defaults to 0.
            train_on_multi_agent (bool, optional): if train on multi-agent
            prediction. If `True`, all predictableDefaults to `True`.
        """
        assert challenge_type in [
            "single-agent",
            "multi-agent",
            "conditional-single-agent",
            "conditional-multi-agent",
        ], ValueError(f"Invalid challenge type {challenge_type:s}.")
        self.challenge_type = challenge_type

        assert split in ["train", "val", "test"], ValueError(
            "Expect tag to be either 'trian', 'val', or 'test', "
            f"but got {split:s}."
        )

        assert isinstance(subsampler, INTERACTIONSubSampler), TypeError(
            "Expect subsampler to be an instance of "
            f"INTERACTIONSubSampler, but got {type(subsampler):s}."
        )
        self.subsampler = subsampler
        self.split = split

        self.radius = radius
        self.force_data_cache = force_data_cache
        self.train_on_multi_agent = train_on_multi_agent

        if isinstance(transform, Iterable):
            transform = Compose(list(transform))

        self._map_api_container, self._track_api_container = {}, {}
        self._indexer = []
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=None,
            pre_filter=None,
        )

        self._load_to_mem()

        if self.force_data_cache:
            raise NotImplementedError("Data Cache not implemented yet.")

    @property
    def raw_file_names(self) -> List[str]:
        """List[str]: list of raw file names."""
        return (
            self._get_raw_map_file_paths() + self._get_raw_track_file_paths()
        )

    @property
    def processed_file_names(self) -> List[str]:
        """List[str]: list of processed file names."""
        return (
            self._get_processed_map_file_paths()
            + self._get_processed_track_file_paths()
        )

    @property
    def cache_dir(self) -> str:
        """str: absolute path of the cache directory."""
        return str(
            Path(self.processed_dir, "cache", self.challenge_type).resolve()
        )

    @property
    def map_root(self) -> str:
        """str: absolute root directory of all the map data files."""
        return str(Path(self.raw_dir, "maps").resolve())

    @property
    def track_root(self) -> str:
        """str: absolute root directory of all the track data files."""
        return str(Path(self.raw_dir, self.tag).resolve())

    @property
    def locations(self) -> List[str]:
        """List[str]: list of locations in the dataset."""
        if self.subsampler.locations is None:
            return SPLITS[self.tag]
        else:
            return [
                name
                for name in SPLITS[self.tag]
                if name in self.subsampler.locations
            ]

    @property
    def tag(self) -> str:
        """str: Tag of the dataset. See official website for details."""
        if self.split in ["train", "val"]:
            return self.split
        return f"{self.split}_{self.challenge_type}"

    def download(self) -> None:
        """Download the INTERACTION dataset to the `self.raw_dir` folder.

        Raises:
            RuntimeError: if the raw data files are not found or incomplete.
        """
        raise RuntimeError(
            "Raw data files not found or incomplete at"
            f" '{self.raw_dir:s}'! "
            "Please visit 'https: // interaction-dataset.com', "
            "download and validate all the data files."
        )

    def get(self, idx: int) -> INTERACTIONData:
        """Get the data object at the given index.

        Args:
            idx (int): index of the data object.

        Returns:
            INTERACTIONData: the data object at the given index.
        """
        location, case_id, ego_id = self._indexer[idx]
        map_api = self._map_api_container[location]
        case_api = self._track_api_container[location][case_id]

        if self.force_data_cache:
            # TODO: implement case caching.
            raise NotImplementedError("Case caching not implemented yet.")
        else:
            data = self._get_case_data(
                map_api=map_api, case_api=case_api, ego_id=int(ego_id)
            )
        data.sample_idx = torch.tensor([idx], dtype=torch.long)
        data.num_data = self.len()

        return data

    def len(self) -> int:
        """Return the number of data in the dataset."""
        return len(self._indexer)

    def process(self) -> None:
        queue = Queue()
        processes: List[Process] = []
        for location in self.locations:
            proc = Process(
                target=self._parse_api,
                args=(
                    queue,
                    location,
                ),
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()

        if not queue.empty():
            error_, traceback_ = queue.get()
            raise RuntimeError(error_)

    # ----------- private methods ------------ #
    def _get_raw_map_file_paths(self) -> List[Path]:
        """Return a list of absolute paths of the raw map data files."""
        return [
            Path(self.map_root, f"{location}.osm").resolve()
            for location in self.locations
        ]

    def _get_processed_map_file_path(self, location: str) -> Path:
        """Return the absolute path of the processed map data file."""
        map_dir = Path(self.processed_dir, "maps")
        if not map_dir.is_dir():
            map_dir.mkdir(parents=True, exist_ok=True)

        return Path(map_dir, f"{location}.gz").resolve()

    def _get_processed_map_file_paths(self) -> List[Path]:
        """Return a list of absolute paths of the processed map data files."""
        return [
            self._get_processed_map_file_path(location)
            for location in self.locations
        ]

    def _get_raw_track_file_paths(self) -> List[Path]:
        if "test" in self.split:
            suffix = "obs"
        else:
            suffix = self.split

        return [
            Path(self.track_root, f"{location}_{suffix}.csv").resolve()
            for location in self.locations
        ]

    def _get_processed_track_file_path(self, location: str) -> Path:
        track_dir = Path(self.processed_dir, "tracks")
        if not track_dir.is_dir():
            track_dir.mkdir(parents=True, exist_ok=True)

        return Path(track_dir, f"{location}_{self.tag}.gz").resolve()

    def _get_processed_track_file_paths(self) -> List[Path]:
        return [
            self._get_processed_track_file_path(location)
            for location in self.locations
        ]

    def _get_case_map(
        self, map_api: INTERACTIONMap, anchor: Tuple[float, float]
    ) -> Dict[str, Tensor]:
        assert isinstance(map_api, INTERACTIONMap), "Invalid map api."

        def _parse_map_features(
            record: pd.Series,
        ) -> Tuple[npt.NDArray, npt.NDArray]:
            obj_id = record.name

            obj_geom: BaseGeometry = record["geometry"]
            if isinstance(obj_geom, Point):
                # single-point polyline
                x, y = obj_geom.xy[0][0], obj_geom.xy[1][0]
                feats = np.array([x, y], dtype=np.float64)
            elif isinstance(obj_geom, LineString):
                feats = np.vstack(obj_geom.coords)
            elif isinstance(obj_geom, MultiLineString):
                xs, ys = [], []
                for line in obj_geom.geoms:
                    xs.extend(line.xy[0])
                    ys.extend(line.xy[1])
                feats = np.vstack([xs, ys]).T
            else:
                raise NotImplementedError(
                    f"Unsupported geometry type: {obj_geom.type:s}."
                )
            try:
                centroid = np.vstack(obj_geom.centroid.coords)
                centroid = np.broadcast_to(centroid, (feats.shape[0], 2))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to get centroid of geometry {obj_geom}: {e}"
                )

            # encode map polyline type
            if record["type"] in (
                "CURBSTONE_LOW",
                "GUARD_RAIL",
                "ROAD_BORDER",
            ):
                type_encoding = np.array([0, 1, 0, 0, 0, 0, 0, 0])
            elif record["type"] == "LINE_THIN_DASHED":
                type_encoding = np.array([0, 0, 1, 0, 0, 0, 0, 0])
            elif record["type"] in (
                "LINE_THIN_SOLID",
                "LINE_THIN_SOLID_SOLID",
            ):
                type_encoding = np.array([0, 0, 0, 1, 0, 0, 0, 0])
            elif record["type"] in (
                "LINE_THICK_SOLID",
                "LINE_THICK_SOLID_SOLID",
            ):
                type_encoding = np.array([0, 0, 0, 0, 1, 0, 0, 0])
            elif record["type"] == "STOP_LINE":
                type_encoding = np.array([0, 0, 0, 0, 0, 1, 0, 0])
            elif record["type"] == "PEDESTRIAN_MARKING":
                type_encoding = np.array([0, 0, 0, 0, 0, 0, 1, 0])
            elif record["type"] in ("VIRTUAL", "VIRTUAL_SOLID"):
                type_encoding = np.array([0, 0, 0, 0, 0, 0, 0, 1])
            type_encoding = np.broadcast_to(
                type_encoding[None, :],
                (feats.shape[0], len(type_encoding)),
            )
            centroid = np.vstack(obj_geom.centroid.coords)

            feats = np.hstack(
                (
                    feats,
                    type_encoding,
                    np.full((feats.shape[0], 1), fill_value=obj_id),
                )
            )

            return feats, centroid

        # parse node features from map layers
        waylayer = map_api.get_map_layer("way")
        waylayer = waylayer[waylayer["type"].isin(WAY_TYPE_OF_INTEREST)]
        # optionally create observation range buffer and clip observations
        if self.radius is not None:
            if self.radius == 0:
                # if radius is 0, return empty map features
                return {
                    "x_map": torch.zeros((0, 8), dtype=torch.float32),
                    "map_clusters": torch.zeros((0,), dtype=torch.long),
                    "map_centroid": torch.zeros((0, 2), dtype=torch.float32),
                }
            buffer: Polygon = Point(anchor).buffer(self.radius)
            waylayer = gpd.clip(waylayer, buffer)
        map_id_mapper = {
            _id: ind
            for ind, _id in enumerate(waylayer.index.unique().tolist())
        }
        waylayer.index = waylayer.index.map(map_id_mapper)

        # parse node features
        _way_outputs = waylayer.apply(_parse_map_features, axis=1)
        node_feats = np.vstack([out[0] for out in _way_outputs])
        node_centroids = np.vstack([out[1] for out in _way_outputs])

        x_map = torch.from_numpy(node_feats).float()
        map_clusters = torch.from_numpy(node_feats[:, -1]).long()
        map_centroid = torch.from_numpy(node_centroids).float()

        return {
            "x_map": x_map,
            "map_clusters": map_clusters,
            "map_centroid": map_centroid,
        }

    def _get_case_tracks(
        self, case_api: INTERACTIONCase, anchor: Tuple[float, float]
    ) -> Dict[str, Tensor]:
        # parse history observation
        hist_df = case_api.history_frame.fillna(0.0)
        # NOTE: change the following lines for different motion features
        cols = ["x", "y", "psi_rad", "vx", "vy", "length", "width"]
        track_ids = torch.from_numpy(hist_df.index.unique().values).long()
        track_pos = torch.from_numpy(
            hist_df.reset_index(drop=False)
            .groupby("track_id")
            .agg({"x": "last", "y": "last"})
            .values
        ).float()

        motion_clusters = (
            hist_df.index.to_series()
            .apply(lambda x: track_ids.tolist().index(x))
            .values
        )
        motion_data = torch.from_numpy(
            np.hstack(
                [
                    hist_df.loc[:, cols].values,
                    motion_clusters[..., None],
                ]
            )
        ).float()
        motion_clusters = torch.from_numpy(motion_clusters).long()
        motion_timestamps = torch.from_numpy(
            hist_df.loc[:, "timestamp_ms"].values
        ).long()

        # get target motion states
        future_df = case_api.futural_frame
        future_df = future_df.loc[
            future_df.index.isin(case_api.tracks_to_predict)
        ]
        if len(future_df) > 0:
            # for training and validation set, there are target motion states
            motion_tar = torch.from_numpy(
                future_df.loc[:, ["x", "y"]].values
            ).float()
            motion_tar_cluster = torch.from_numpy(
                future_df.index.values
            ).long()
            motion_tar_valid = ~torch.from_numpy(
                future_df.loc[:, cols].isna().values
            ).bool()
        else:
            # for test set, there are no target motion states
            motion_tar = torch.zeros((0, len(cols)), dtype=torch.float32)
            motion_tar_cluster = torch.zeros((0,), dtype=torch.long)
            motion_tar_valid = torch.zeros((0, len(cols)), dtype=torch.bool)

        return {
            "motion_data": motion_data,
            "motion_clusters": motion_clusters,
            "motion_timestamps": motion_timestamps,
            "track_ids": track_ids,
            "track_pos": track_pos,
            "motion_tar": motion_tar,
            "motion_tar_valid": motion_tar_valid,
            "motion_tar_cluster": motion_tar_cluster,
        }

    def _get_case_data(
        self,
        map_api: INTERACTIONMap,
        case_api: INTERACTIONCase,
        ego_id: Optional[int | Iterable[int]] = None,
    ) -> INTERACTIONData:
        if ego_id is None:
            ego_id = case_api.current_frame.index.tolist()
        if isinstance(ego_id, Iterable):
            anchor: npt.NDArray = case_api.current_frame.loc[
                case_api.current_frame.index.isin(ego_id),
                ["x", "y", "psi_rad"],
            ].values.mean(0)
            anchor = anchor.astype(float).mean(0)
        else:
            anchor = case_api.current_frame.loc[ego_id, ["x", "y", "psi_rad"]]
            anchor: npt.NDArray = anchor.values.astype(float)
            ego_id = [int(ego_id)]
        map_data = self._get_case_map(map_api, tuple(anchor[0:2]))
        tracks_data = self._get_case_tracks(case_api, tuple(anchor[0:2]))
        anchor = torch.tensor(anchor, dtype=torch.float32)

        # NOTE: uncomment following lines to train on multi-agent prediction
        y_motion = tracks_data["motion_tar"]
        y_valid = tracks_data["motion_tar_valid"]
        y_cluster = tracks_data["motion_tar_cluster"]
        if self.split == "train" and self.train_on_multi_agent:
            tracks_to_predict = case_api.tracks_to_predict
            tracks_to_predict = torch.tensor(
                tracks_to_predict, dtype=torch.long
            )
        else:
            tracks_to_predict = torch.tensor(data=ego_id, dtype=torch.long)
            if self.split != "test":
                tar_filter = torch.isin(
                    tracks_data["motion_tar_cluster"], tracks_to_predict
                )
                y_motion = y_motion[tar_filter]
                y_valid = y_valid[tar_filter]
                y_cluster = y_cluster[tar_filter]

        return INTERACTIONData(
            x_map=map_data["x_map"],
            map_clusters=map_data["map_clusters"],
            map_centroid=map_data["map_centroid"],
            x_motion=tracks_data["motion_data"],
            motion_cluster=tracks_data["motion_clusters"],
            motion_timestamps=tracks_data["motion_timestamps"],
            track_ids=tracks_data["track_ids"],
            track_pos=tracks_data["track_pos"],
            y_motion=y_motion,
            y_valid=y_valid,
            y_cluster=y_cluster,
            anchor=anchor,
            tracks_to_predict=tracks_to_predict,
            num_agents=torch.tensor([case_api.num_agents], dtype=torch.long),
        )

    def _load_to_mem(self) -> None:
        """Load all intermediate objects to memory."""
        LOGGER.info(f"Loading {self.tag} scenarios...")
        for location in self.locations:
            with gzip.open(
                self._get_processed_map_file_path(location), "rb"
            ) as file:
                map_api: INTERACTIONMap = pickle.load(file)
            with gzip.open(
                self._get_processed_track_file_path(location), "rb"
            ) as file:
                track_api: INTERACTIONScenario = pickle.load(file)

            self._map_api_container[location] = map_api
            self._track_api_container[location] = track_api

            if self.subsampler.ratio < 1.0:
                k = track_api.num_cases * self.subsampler.ratio
                self._indexer.extend(
                    # single-agent prediction
                    [
                        (track_api.location, case_id, track_id)
                        for (case_id, track_ids) in random.sample(
                            track_api._tracks_to_predict.items(), k=int(k)
                        )
                        for track_id in track_ids
                    ]
                    if "single" in self.challenge_type
                    # multi-agent prediction
                    else [
                        (track_api.location, case_id, track_ids)
                        for (case_id, track_ids) in random.sample(
                            track_api._tracks_to_predict.items(), k=int(k)
                        )
                    ]
                )

            else:
                self._indexer.extend(
                    # single-agent prediction
                    [
                        (track_api.location, case_id, track_id)
                        for (
                            case_id,
                            track_ids,
                        ) in track_api._tracks_to_predict.items()
                        for track_id in track_ids
                    ]
                    if "single" in self.challenge_type
                    # multi-agent prediction
                    else [
                        (track_api.location, case_id, track_ids)
                        for (
                            case_id,
                            track_ids,
                        ) in track_api._tracks_to_predict.items()
                    ]
                )

        LOGGER.info("Loading scenarios for dataset...DONE!")

    def _parse_api(self, queue: Queue, location: str) -> None:
        """Parse scenario APIs and save to cache."""
        try:
            LOGGER.info("Processing scenario %s...", location)
            map_filepath = self._get_processed_map_file_path(location)
            if not map_filepath.is_file():
                # process and cache map data
                map_api = INTERACTIONMap(root=self.map_root, location=location)
                with gzip.open(map_filepath, mode="wb") as file:
                    pickle.dump(map_api, file)

            track_filepath = self._get_processed_track_file_path(location)
            if not track_filepath.is_file():
                # process and cache scenario api
                track_api = INTERACTIONScenario(
                    root=self.track_root, location=location
                )
                with gzip.open(track_filepath, mode="wb") as file:
                    pickle.dump(track_api, file)
            LOGGER.info("Processing scenario %s...DONE!", location)
        except Exception as err:
            tb = traceback.format_exc()
            queue.put((err, tb))

    # TODO (Juanwu): add support for data caching

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__}(num_data={self.len()})"
            f" at {hex(id(self))}>"
        )

    def __repr__(self) -> str:
        return str(self)
