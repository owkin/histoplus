"""Tiles Map Dataset. This is a merge of the TilingTool's TilesMap and the ClassicAlgos's TilesMap."""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from torch.utils.data import Dataset


class TilesMap(Dataset):
    """
    Tiles Map Dataset.

    From a slide given by slide_path, creates a map-style PyTorch Dataset object
    that returns n_samples tiles sampled, either randomly or
    sequentially from the matter.

    .. note::
        It is generably preferable to use a map style Dataset instead of an IterableDataset. BUT,
        on some cases, mounting the slide to the /dev/shm RAM volume will speed up the read region process.
        To use this functionality, you should use TilesIterator instead.

    This dataset returns PIL.Image samples. In order to be used in addition with
    a `torch.utils.data.DataLoader`, you must transforms the samples to `torch.Tensor`.
    `torchvision.transforms.ToTensor` can by use for this purpose.

    Example:
        >>> from torchvision.transforms import ToTensor
        >>> iterator = TilesMap(..., transform=ToTensor())

    .. warning::
        You must install Openslide and the histo extra requirements before using this Class.

        >>> apt-get update -qq && apt-get install openslide-tools libgeos-dev -y 2>&1
        >>> pip install "classic-algos[histo]"

    Parameters
    ----------
    slide: OpenSlideType
        Slide to extract tiles from
    deepzoom: DeepZoomType
        Use a DeepZoomGenerator instance to sample the tiles with its `get_tile` method
    tissue_coords: numpy.ndarray
        If you are using the TilingTool coordinates, this are the 2:3 dimensions.
    tiles_level: int
        level to extract the tiles from. If you are using the TilingTool coordinates,
        this is the first dimension.
    tiles_size: int
        size of the tiles. The returned tiles will be PIL.Image
        object of size (tiles_size, tiles_size, 3)
    n_samples: Optional[int] = None
        number of tiles to sample
    random_sampling: bool = False
        sample either randomly or sequentially.
    transform: Optional[Callable] = None
        transformation to apply to the PIL.Image after sampling
    metadata: bool = False
        If true, returns a Tuple with a dict containing the metadata of the slide.
        Useful for tiling.
    float_coords: bool = False
        Whether to enable float coordinates for deepzoom. This enables tile sampling
        outside of the deepzoom generated coordinates grid.
    """

    def __init__(
        self,
        slide: OpenSlide,
        deepzoom: DeepZoomGenerator,
        tissue_coords: np.ndarray,
        tiles_level: int,
        tiles_size: int = 224,
        n_samples: Optional[int] = None,
        random_sampling: bool = False,
        transform: Optional[Callable] = None,
        metadata: bool = False,
        float_coords: bool = False,
    ):
        if n_samples is None:
            n_samples = len(tissue_coords)
        else:
            n_samples = min(n_samples, len(tissue_coords))

        if tissue_coords.ndim == 3:
            raise ValueError("tissue_coords must have only two dimensions.")

        self.n_samples = n_samples
        self.slide_path = Path(str(slide._filename))
        self.transform = transform
        self._tiles_level = tiles_level
        self._tissue_coords = tissue_coords
        if not float_coords:
            self._tissue_coords = self._tissue_coords.astype(int)
        self._tiles_size = [tiles_size, tiles_size]
        self._metadata = metadata

        self.wsi = slide

        self.dz = deepzoom

        self.random_sampling = random_sampling

    def sample_new_tiles(self) -> None:
        """Permute tile indices to sample new tiles.

        Should be called at the end of every epoch.

        Raises
        ------
        ValueError: samples_new_tiles should only be called if random_sampling is set.
        """
        if not self.random_sampling:
            raise ValueError(
                "samples_new_tiles should only be called if random_sampling is set."
            )

        indices = np.random.permutation(np.arange(0, len(self._tissue_coords)))
        self.indices = indices[: self.n_samples]

    @property
    def random_sampling(self):
        """Whether the tiles are sampled randomly or sequentially."""
        return self._random_sampling

    @random_sampling.setter
    def random_sampling(self, value: bool):
        self._random_sampling = value

        if value:
            self.sample_new_tiles()
        else:
            self.indices = np.arange(0, len(self._tissue_coords))[: self.n_samples]

    def __getitem__(self, item: int):
        """Get the item at the given index.

        Returns
        -------
        image: PIL.Image
            Image.
        """
        # True index of the tile. If random_sampling is False, same index.
        index = self.indices[item]
        coords = self._tissue_coords[index, :]

        if self.dz is not None:
            tile = self.dz.get_tile(
                level=int(self._tiles_level),
                address=(coords[0], coords[1]),
            )
        else:
            tile = self.wsi.read_region(
                coords, int(self._tiles_level), self._tiles_size
            ).convert("RGB")

        if tile.size != self._tiles_size:
            # if tile is on a border, we need to pad it
            tile_arr = np.array(tile)
            tile_arr = np.pad(
                tile_arr,
                pad_width=(
                    (0, self._tiles_size[0] - tile_arr.shape[0]),
                    (0, self._tiles_size[1] - tile_arr.shape[1]),
                    (0, 0),
                ),
            )
            tile = Image.fromarray(tile_arr)

        if self.transform:
            tile = self.transform(tile)

        if self._metadata:
            return (
                tile,
                {
                    "slide_name": self.slide_path.name,
                    "coords": coords,
                    "level": self._tiles_level,
                    "slide_length": len(self),
                },
            )
        else:
            return tile

    def __len__(self):
        return self.n_samples
