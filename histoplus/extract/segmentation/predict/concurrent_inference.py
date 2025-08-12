"""Module for concurrent inference of segmentation models.

This module provides a class for running segmentation model inference with concurrent
compression and saving of results.

It uses a thread pool to compress and save the results as they are generated, allowing
for efficient parallel processing.
"""

import os
import queue
from multiprocessing import Process, Queue
from typing import Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from histoplus.helpers.segmentor.base import Segmentor


class ConcurrentModelInference:
    """Concurrent model inference handler.

    Parameters
    ----------
    model: Segmentor
        The PyTorch model for inference (should be on GPU)

    output_dir: str
        Directory to save compressed output arrays

    queue_size: int
        Maximum size of the task queue

    max_workers: int
        Maximum number of worker processes for compression/saving
    """

    def __init__(
        self,
        model: Segmentor,
        output_dir: str,
        queue_size: int = 100,
        max_workers: int = 4,
        verbose: int = 1,
    ):
        self._model = model
        self._output_dir = output_dir
        self._queue_size = queue_size
        self._max_workers = max_workers
        self._verbose = verbose

        self._queue: Queue[Optional[tuple[dict[str, np.ndarray], int]]] = Queue(
            maxsize=queue_size
        )
        self._workers = [
            Process(
                target=self._compression_worker, args=(self._queue, self._output_dir)
            )
            for _ in range(self._max_workers)
        ]

        os.makedirs(output_dir, exist_ok=True)

        for worker in self._workers:
            worker.start()

    def _compression_worker(self, task_queue: Queue, output_dir: str):
        """Worker process that continuously pulls compression tasks from the queue."""
        while True:
            try:
                data = task_queue.get(timeout=1)
                if data is None:
                    break

                batch_predictions, batch_idx = data
                self._compress_and_save(batch_predictions, output_dir, batch_idx)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in compression worker: {e}")

    @staticmethod
    def _compress_and_save(
        outputs: dict[str, np.ndarray],
        predictions_dir: str,
        batch_idx: int,
    ):
        """Compress numpy array and save to disk using zlib."""
        batch_predictions_path = os.path.join(
            predictions_dir, f"batch_{batch_idx:05d}.npz"
        )
        np.savez_compressed(batch_predictions_path, **outputs)  # type: ignore

    def run(self, dataloader: DataLoader):
        """Run inference on data loader and concurrently save compressed results.

        Parameters
        ----------
        dataloader: DataLoader
            Iterator that yields batches
        """
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="Inference",
            leave=False,
            disable=bool(self._verbose == 0),
        )

        try:
            with torch.inference_mode():
                for batch_idx, images in pbar:
                    batch_predictions = self._model.forward(images)

                    np_batch_predictions = {
                        key: predictions.detach().cpu().numpy()
                        for key, predictions in batch_predictions.items()
                    }
                    del batch_predictions

                    self._queue.put((np_batch_predictions, batch_idx))

        finally:
            for _ in range(self._max_workers):
                self._queue.put(None)

            for w in self._workers:
                w.join()

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            for _ in range(self._max_workers):
                self._queue.put(None)

            for w in self._workers:
                if w.is_alive():
                    w.join(timeout=1.0)
                    if w.is_alive():
                        w.terminate()
        except Exception:
            pass
