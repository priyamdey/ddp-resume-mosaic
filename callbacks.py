from __future__ import annotations

import logging
import os
import pathlib
import tempfile
import textwrap
from typing import Callable, List, Optional, Tuple, Union

from composer.core import Event, State
from composer.core.callback import Callback
from composer.core.time import Time, Timestamp, TimeUnit
from composer.loggers import Logger
from composer.loggers.logger import LogLevel
from composer.utils import checkpoint, dist, is_model_deepspeed
from composer.utils.file_helpers import (
    FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, FORMAT_NAME_WITH_DIST_TABLE,
    create_symlink_file, ensure_folder_has_no_conflicting_files,
    format_name_with_dist, format_name_with_dist_and_time, is_tar)

from head_adder import HeadAdder

log = logging.getLogger(__name__)

__all__ = ['CheckpointSaver', 'checkpoint_periodically']


def checkpoint_periodically(interval: Union[str, int, Time]) -> Callable[[State, Event], bool]:
    if isinstance(interval, str):
        interval = Time.from_timestring(interval)
    if isinstance(interval, int):
        interval = Time(interval, TimeUnit.EPOCH)

    if interval.unit == TimeUnit.EPOCH:
        save_event = Event.EPOCH_CHECKPOINT
    elif interval.unit == TimeUnit.BATCH:
        save_event = Event.BATCH_CHECKPOINT
    else:
        raise NotImplementedError(
            f'Unknown checkpointing interval: {interval.unit}. Must be TimeUnit.EPOCH or TimeUnit.BATCH.')

    last_checkpoint_batch: Optional[Time] = None

    def save_interval(state: State, event: Event):
        nonlocal last_checkpoint_batch
        elapsed_duration = state.get_elapsed_duration()
        assert elapsed_duration is not None, 'elapsed_duration is set on the BATCH_CHECKPOINT and EPOCH_CHECKPOINT'

        if elapsed_duration >= 1.0:
            # if doing batch-wise checkpointing, and we saved a checkpoint at the batch_checkpoint event
            # right before the epoch_checkpoint event, do not save another checkpoint at the epoch_checkpoint
            # event if the batch count didn't increase.
            if state.timestamp.batch != last_checkpoint_batch:
                last_checkpoint_batch = state.timestamp.batch
                return True

        if event == Event.EPOCH_CHECKPOINT:
            for algo in state.algorithms:
                if isinstance(algo, HeadAdder):
                    if state.timestamp.epoch == algo.surgery_time:
                        print(f'\nSAVING CKPT BEFORE HEAD ADDITION at epoch {algo.surgery_time}\n')
                        return True

        if save_event == Event.EPOCH_CHECKPOINT:
            count = state.timestamp.epoch
        elif save_event == Event.BATCH_CHECKPOINT:
            count = state.timestamp.batch
        else:
            raise RuntimeError(f'Invalid save_event: {save_event}')

        if event == save_event and int(count) % int(interval) == 0:
            last_checkpoint_batch = state.timestamp.batch
            return True

        return False

    return save_interval


class CheckpointSaver(Callback):
    def __init__(
        self,
        folder: str = '{run_name}/checkpoints',
        filename: str = 'ep{epoch}-ba{batch}-rank{rank}.pt',
        artifact_name: Optional[str] = '{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}',
        latest_filename: Optional[str] = 'latest-rank{rank}.pt',
        latest_artifact_name: Optional[str] = '{run_name}/checkpoints/latest-rank{rank}',
        save_interval: Union[Time, str, int, Callable[[State, Event], bool]] = '1ep',
        *,
        overwrite: bool = False,
        num_checkpoints_to_keep: int = -1,
        weights_only: bool = False,
    ):
        if not callable(save_interval):
            save_interval = checkpoint_periodically(save_interval)
        self.folder = folder
        self.filename = filename
        self.artifact_name = artifact_name
        self.latest_filename = latest_filename
        self.latest_artifact_name = latest_artifact_name
        self.overwrite = overwrite

        self.save_interval = save_interval
        self.saved_checkpoints: List[Tuple[Timestamp, List[pathlib.Path]]] = []
        self.num_checkpoints_to_keep = num_checkpoints_to_keep
        self.weights_only = weights_only

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused
        folder = format_name_with_dist(self.folder, state.run_name)
        print(f'CHECKPOINT FOLDER: {folder}')
        os.makedirs(folder, exist_ok=True)

    def fit_start(self, state: State, logger: Logger) -> None:
        del logger  # unused
        # Verify safety with self.overwrite. Note that this has to be done at fit_start as opposed to init since it requires state.timestamp
        # from any checkpoints which are loaded, and checkpoint loading happens after Event.INIT.
        if not self.overwrite:
            folder = format_name_with_dist(self.folder, state.run_name)
            ensure_folder_has_no_conflicting_files(folder, self.filename, state.timestamp)
        # Ensure no rank proceeds (and potentially attempts to write to the folder), until all ranks have validated that the folder is safe.
        dist.barrier()
        if is_model_deepspeed(state.model):
            if self.weights_only:
                NotImplementedError(
                    ('Saving checkpoints with `weights_only=True` is not currently supported when using DeepSpeed. '
                     'See https://github.com/mosaicml/composer/issues/685.'))

    def batch_checkpoint(self, state: State, logger: Logger):
        if self.save_interval(state, Event.BATCH_CHECKPOINT):
            # If training is finished, log at the FIT loglevel
            elapsed_duration = state.get_elapsed_duration()
            assert elapsed_duration is not None, 'elapsed_duration is set on Event.BATCH_CHECKPOINT'
            log_level = LogLevel.BATCH if elapsed_duration < 1.0 else LogLevel.FIT
            self._save_checkpoint(state, logger, log_level)

    def epoch_checkpoint(self, state: State, logger: Logger):
        if self.save_interval(state, Event.EPOCH_CHECKPOINT):
            elapsed_duration = state.get_elapsed_duration()
            assert elapsed_duration is not None, 'elapsed_duration is set on Event.BATCH_CHECKPOINT'
            log_level = LogLevel.EPOCH if elapsed_duration < 1.0 else LogLevel.FIT
            self._save_checkpoint(state, logger, log_level)

    def _save_checkpoint(self, state: State, logger: Logger, log_level: LogLevel):
        checkpoint_filepath = os.path.join(format_name_with_dist(self.folder, state.run_name), self.filename)
        checkpoint_filepaths = checkpoint.save_checkpoint(state, checkpoint_filepath, weights_only=self.weights_only)

        if dist.get_global_rank() < len(checkpoint_filepaths):
            # Log the checkpoint as an artifact
            checkpoint_filepath = checkpoint_filepaths[dist.get_global_rank()]
            if self.artifact_name is not None:
                artifact_name = format_name_with_dist_and_time(self.artifact_name, state.run_name,
                                                               state.timestamp).lstrip('/')
                if is_model_deepspeed(state.model) and not is_tar(artifact_name):
                    # Deepspeed requires tarballs; appending `.tar`
                    artifact_name += '.tar'
                logger.file_artifact(log_level=log_level,
                                     artifact_name=artifact_name,
                                     file_path=checkpoint_filepath,
                                     overwrite=self.overwrite)

            if self.latest_filename is not None:
                formatted_folder_path = format_name_with_dist(self.folder, state.run_name)
                symlink_name = os.path.join(
                    formatted_folder_path,
                    format_name_with_dist_and_time(
                        self.latest_filename,
                        state.run_name,
                        state.timestamp,
                    ).lstrip('/'),
                )
                if is_model_deepspeed(state.model) and not is_tar(symlink_name):
                    # Deepspeed requires tarballs; appending `.tar`
                    symlink_name += '.tar'
                symlink_dirname = os.path.dirname(symlink_name)
                if symlink_dirname:
                    os.makedirs(symlink_dirname, exist_ok=True)
                try:
                    os.remove(symlink_name)
                except FileNotFoundError:
                    pass
                relative_checkpoint_path = os.path.relpath(checkpoint_filepath, formatted_folder_path)
                os.symlink(relative_checkpoint_path, symlink_name)
                if self.artifact_name is not None and self.latest_artifact_name is not None:
                    symlink_artifact_name = format_name_with_dist_and_time(self.latest_artifact_name, state.run_name,
                                                                           state.timestamp).lstrip('/') + '.symlink'
                    artifact_name = format_name_with_dist_and_time(self.artifact_name, state.run_name,
                                                                   state.timestamp).lstrip('/')
                    # Always overwrite for symlinks since we use the same filename for latest
                    with tempfile.TemporaryDirectory() as tmpdir:
                        symlink_filename = os.path.join(tmpdir, 'latest.symlink')
                        create_symlink_file(artifact_name, symlink_filename)
                        logger.file_artifact(
                            log_level=log_level,
                            artifact_name=symlink_artifact_name,
                            file_path=symlink_filename,
                            overwrite=True,
                        )

        timestamp = state.timestamp

        pre_surgery_ckpt = False
        for algo in state.algorithms:
            if isinstance(algo, HeadAdder):
                if state.timestamp.epoch == algo.surgery_time:
                    pre_surgery_ckpt = True

        if not pre_surgery_ckpt:
            self.saved_checkpoints.append((timestamp, checkpoint_filepaths))
            if self.num_checkpoints_to_keep >= 0:
                while len(self.saved_checkpoints) > self.num_checkpoints_to_keep:

                    timestamp, checkpoint_filepaths = self.saved_checkpoints[0]
                    if dist.get_global_rank() < len(checkpoint_filepaths):
                        # Remove this rank's checkpoint
                        os.remove(checkpoint_filepaths[dist.get_global_rank()])
                    del self.saved_checkpoints[0]
