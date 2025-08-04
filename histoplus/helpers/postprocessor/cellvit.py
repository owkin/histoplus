"""HoverNet post-processor."""

from typing import Optional

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, label
from skimage.segmentation import watershed

from histoplus.helpers.types import TilePrediction

from .base import Postprocessor


class CellViTPostprocessor(Postprocessor):
    """Post-process the raw predictions of HoVerNet-based models (HoVerNet, CellViT).

    Parameters
    ----------
    cell_type_mapping: dict[int, str]
        Mapping from integer class types to cell type names.
    """

    def __init__(self, mpp: float, cell_type_mapping: dict[int, str]):
        super().__init__()

        self.mpp = mpp
        self.cell_type_mapping = cell_type_mapping

    def postprocess(self, outputs: dict[str, np.ndarray]) -> list[TilePrediction]:
        """Post-process the raw output of the model.

        Parameters
        ----------
        outputs: dict[str, torch.Tensor]
            The raw output of the model.

        Returns
        -------
        list[TilePrediction]
            A list of segmentation masks polygon coordinates.
        """
        np_preds = outputs["np"]
        tp_preds = outputs["tp"]
        hv_preds = outputs["hv"]

        batch_size = np_preds.shape[0]

        predictions = []

        for idx in range(batch_size):
            masks, boxes, centroids, cell_type_ids, probabilities = (
                self._postprocess_output_maps(
                    np_preds[idx], tp_preds[idx], hv_preds[idx]
                )
            )

            centroids = centroids.astype(float)

            mask_coordinates = []
            box_coordinates = []
            idx_valid_masks = []
            for idx_mask, (mask, box) in enumerate(zip(masks, boxes, strict=True)):
                coords = get_polygon_coordinates_from_mask(mask, box)
                if coords is not None:
                    coords = coords.astype(float)
                    mask_coordinates.append(coords)
                    box_coordinates.append(box.astype(float))
                    idx_valid_masks.append(idx_mask)

            valid_masks_array = np.array(idx_valid_masks, dtype=int)

            probabilities = probabilities[valid_masks_array]
            cell_type_ids = cell_type_ids[valid_masks_array]
            centroids = centroids[valid_masks_array]

            cell_types = [
                self.cell_type_mapping[cell_type] for cell_type in cell_type_ids
            ]

            assert len(mask_coordinates) == len(box_coordinates)
            assert len(boxes) == len(centroids)
            assert len(centroids) == len(cell_types)
            assert len(cell_types) == len(probabilities)

            tile_predictions = TilePrediction(
                contours=mask_coordinates,
                bounding_boxes=box_coordinates,
                centroids=centroids.astype(float).tolist(),
                cell_types=cell_types,
                cell_type_probabilities=probabilities.tolist(),
            )

            predictions.append(tile_predictions)

        return predictions

    def _get_postprocessing_hyperparameters(self) -> tuple[int, int]:
        """Get the post-processing hparams based on the MPP used during training.

        Returns
        -------
        tuple[int, int]
            The minimum size to consider a blob an instance.
            The size of the Sobel kernel.
        """
        if self.mpp == 0.5:
            return 5, 11
        if self.mpp == 0.25:
            return 10, 21
        raise ValueError(f"Unknown MPP value. Got {self.mpp}. Expected 0.25 or 0.5")

    def _postprocess_output_maps(  # noqa: PLR0915
        self, np_preds: np.ndarray, tp_preds: np.ndarray, hv_preds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform the NP, TP and HV output maps into a segmentation mask.

        It is modified and simplified to fit our pipeline. Here are the changes:
            1. We remove the centroids generation phase.
            2. Add a `labels` key to compute the F1 score.

        The post-processing is split into two steps:

        1. The NP and HV maps are combined in the `_process_np_hv_maps` method to output
        a segmentation mask.

        2. Using the segmentation mask and the TP map, the detected instances are
        assigned a class.

        These properties are subsequently used to build the InstanceSegmentationLabel
        object.

        Parameters
        ----------
        np_preds: np.ndarray, shape [H, W]
            The NP prediction map.

        tp_preds: np.ndarray, shape [H, W]
            The TP prediction map. Each value is a class of the pixel.

        hv_preds: np.ndarray, shape [2, H, W]
            The HV prediction map.

        Returns
        -------
        masks, np.ndarray, shape [n_instances, H, W]
            Binary segmentation map with n_instances channels.

        boxes, np.ndarray, shape [n_instances, 4]
            The 4 coordinates of the bounding box

        centroids, np.ndarray, shape [n_instances, 2]
            The centroids of the instances.

        labels, np.ndarray, shape [n_instances]
            The classes/labels of the bounding box.

        scores, np.ndarray, shape [n_instances]
            The scores (proba) of the predicted label.
        """
        tp_preds = tp_preds.astype(np.int32)
        np_preds = np_preds.astype(np.float32)
        hv_preds = hv_preds.astype(np.float32)

        pred_type = tp_preds
        pred_inst = self._process_np_hv_maps(np_preds, hv_preds)

        masks, boxes, centroids, labels, scores = [], [], [], [], []

        inst_id_list = np.unique(pred_inst)[1:]  # Exclude background

        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id

            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ]
            inst_map = inst_map.astype(np.uint8)

            inst_moment = cv2.moments(inst_map)

            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour_arr = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour_arr.shape[0] < 3:
                continue
            if len(inst_contour_arr.shape) != 2:
                continue  # ! check for trickery shape

            inst_centroid = np.array(
                [
                    (inst_moment["m10"] / inst_moment["m00"]),
                    (inst_moment["m01"] / inst_moment["m00"]),
                ]
            )
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y

            # Get a class of each instance id, stored at index id-1
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]

            inst_map_crop = inst_map_crop == inst_id
            inst_type = inst_type_crop[inst_map_crop]

            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels, strict=True))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)

            inst_type = type_list[0][0]
            if inst_type == 0:  # if predicted cell type is background
                if len(type_list) > 1:
                    # pick second most likely cell type if it exists
                    inst_type = type_list[1][0]
                else:
                    # otherwise, skip it
                    continue

            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)

            boxes.append([rmin, rmax, cmin, cmax])
            masks.append((pred_inst == inst_id).astype(np.uint8))
            centroids.append(inst_centroid)

            labels.append(int(inst_type))
            scores.append(float(type_prob))

        masks_arr = np.array(masks)
        boxes_arr = np.array(boxes)
        centroids_arr = np.array(centroids)
        labels_arr = np.array(labels)
        scores_arr = np.array(scores)

        assert masks_arr.shape[0] == boxes_arr.shape[0]
        assert masks_arr.shape[0] == centroids_arr.shape[0]
        assert boxes_arr.shape[0] == labels_arr.shape[0]
        assert labels_arr.shape[0] == scores_arr.shape[0]

        return masks_arr, boxes_arr, centroids_arr, labels_arr, scores_arr

    def _process_np_hv_maps(
        self, nuclei_proba: np.ndarray, hv_map: np.ndarray
    ) -> np.ndarray:
        """Process the NP and HV maps and output a segmentation map.

        Parameters
        ----------
        nuclei_proba: np.ndarray, shape [H, W]
            Probability map of being an instance (channel 2 of np_map)

        hv_map: np.ndarray, shape [2, H, W]
            Horizontal and vertical distances array to the center of mass of
            the instances.

        Returns
        -------
        np.ndarray, shape [H, W]
            Segmentation map.
        """
        blb_raw = nuclei_proba
        hv_raw = hv_map

        h_dir_raw = hv_raw[0]
        v_dir_raw = hv_raw[1]

        # hyperparameters
        min_size, kernel_size = self._get_postprocessing_hyperparameters()

        # processing
        blb = np.array(blb_raw >= 0.5, dtype=np.int32)

        blb = label(blb)[0]
        blb = remove_small_objects(blb, min_size=min_size)
        blb[blb > 0] = 1  # background is 0 already

        h_dir = cv2.normalize(  # type: ignore
            h_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(  # type: ignore
            v_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=kernel_size)

        sobelh = 1 - (
            cv2.normalize(  # type: ignore
                sobelh,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )
        sobelv = 1 - (
            cv2.normalize(  # type: ignore
                sobelv,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        dist = (1.0 - overall) * blb
        ## nuclei values form mountains so inverse to get basins
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        overall = np.array(overall >= 0.4, dtype=np.int32)

        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = label(marker)[0]
        marker = remove_small_objects(marker, min_size=min_size)

        proced_pred = watershed(dist, markers=marker, mask=blb)

        return proced_pred


def remove_small_objects(
    pred: np.ndarray, min_size: int = 64, connectivity: int = 1
) -> np.ndarray:
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided.

    Taken from: https://github.com/vqdang/hover_net/blob/master/misc/utils.py#L142

    Parameters
    ----------
    pred: np.ndarray
        Input array.
    min_size: int
        Minimum size of instance in output array
    connectivity: int
        The connectivity defining the neighborhood of a pixel.

    Returns
    -------
        out: output array with instances removed under min_size
    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    component_sizes = np.bincount(ccs.ravel())

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def get_bounding_box(img: np.ndarray) -> list[int]:
    """Get bounding box coordinate information.

    Taken from: https://github.com/vqdang/hover_net/blob/master/misc/utils.py#L18

    Parameters
    ----------
    img : np.ndarray
        The region.

    Returns
    -------
    tuple[int, int, int, int]
        Bounding box coordinates.
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max else accessing will be 1px in the
    # box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def get_polygon_coordinates_from_mask(
    mask: np.ndarray, box: np.ndarray
) -> Optional[np.ndarray]:
    """Get polygon coordinates from mask.

    Parameters
    ----------
    mask : np.ndarray, shape [H, W]
        The mask.
    box : np.ndarray, shape [4]
        Bounding box coordinates.

    Returns
    -------
    np.ndarray, shape [n_points, 2]
        The polygon coordinates.
    """
    y_min, y_max, x_min, x_max = box.astype(int)

    # Take a wider region around the box
    y_min = int(max(y_min - 2, 0))
    x_min = int(max(x_min - 2, 0))
    x_max = int(min(x_max + 2, mask.shape[1] - 1))
    y_max = int(min(y_max + 2, mask.shape[0] - 1))

    mask_crop = mask[y_min:y_max, x_min:x_max]

    contours_crop = cv2.findContours(mask_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_crop[0]) == 0:
        # If no contour is detected, return None
        return None

    # only has 1 instance per map, no need to check #contour detected by opencv
    contours_crop_array = np.squeeze(
        contours_crop[0][0].astype("int32"),  # type: ignore
        axis=1,  # type: ignore
    )
    contours_crop_array += np.array([[x_min, y_min]])  # index correction

    return contours_crop_array


def extract_centroid(mask: np.ndarray) -> np.ndarray:
    """Compute centroid from mask of one individual cell.

    Parameters
    ----------
    mask: np.ndarray
        Mask of size [height, width] where pixels are either 1 or 0.

    Returns
    -------
    centroid: np.ndarray
        of shape (2,) : [x, y].
    """
    # warning ! [y,x] is in the image convention, y is the first dimension and x the
    # second.
    where_mask_1 = np.argwhere(mask == 1)
    if len(where_mask_1) == 0:
        return np.array([np.nan, np.nan])
    centroid = where_mask_1.mean(axis=0)[::-1]
    return centroid
