# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Useful functions for dealing with streaming datasets."""
import autoroot 
import autorootcwd
import os
from pathlib import Path
from typing import Sequence

from streaming import Stream


def make_streams(remote, local=None, proportion=None, repeat=None, choose=None):
    """Helper function to create a list of Stream objects from a set of remotes and stream weights.

    Args:
        remote (Union[str, Sequence[str]]): The remote path or paths to stream from.
        local (Union[str, Sequence[str]], optional): The local path or paths to cache the data.
            If not provided, the default local path is used (subdirectories under
            `os.getenv("TMP_DIR", "/tmp")` matching the remote path structure).
            Defaults to ``None``.
        proportion (list, optional): Specifies how to sample this Stream relative to other Streams.
            If None, all streams are sampled equally. Must be the same length as `remote`.
            Defaults to ``None``.
        repeat (list, optional): Specifies the degree to which a Stream is upsampled or downsampled.
            If None, no up/downsampling. Must be the same length as `remote`.
            Defaults to ``None``.
        choose (list, optional): Specifies the number of samples to choose from a Stream.
            If None, all samples are chosen. Must be the same length as `remote`.
            Defaults to ``None``.

    Returns:
        List[Stream]: A list of Stream objects configured according to the inputs.
    """
    remote, local = _make_remote_and_local_sequences(remote, local)
    proportion, repeat, choose = _make_weighting_sequences(remote, proportion, repeat, choose)

    streams = []
    for i, (r, l) in enumerate(zip(remote, local)):
        streams.append(Stream(remote=r, local=l, proportion=proportion[i], repeat=repeat[i], choose=choose[i]))
    return streams


def _make_remote_and_local_sequences(remote, local=None):
    """Converts remote and local paths to sequences and validates them.

    If `remote` or `local` are strings, they are converted to single-element lists.
    If `local` is not provided, default local paths are generated based on `remote` paths.
    Ensures that `remote` and `local` sequences have the same length.

    Args:
        remote (Union[str, Sequence[str]]): Remote path(s).
        local (Union[str, Sequence[str]], optional): Local path(s). Defaults to None.

    Returns:
        tuple[Sequence[str], Sequence[str]]: A tuple containing the sequence of remote paths
            and the sequence of local paths.

    Raises:
        ValueError: If `remote` and `local` are provided and have different lengths,
            or if their types are inconsistent (not both strings or both sequences).
    """
    if isinstance(remote, str):
        remote = [remote]
    if isinstance(local, str):
        local = [local]
    if not local:
        local = [_make_default_local_path(r) for r in remote]

    if isinstance(remote, Sequence) and isinstance(local, Sequence):
        if len(remote) != len(local):
            raise ValueError(f"remote and local Sequences must be the same length, got lengths {len(remote)} and {len(local)}")
    else:
        raise ValueError(f"remote and local must be both Strings or Sequences, got types {type(remote)} and {type(local)}.")
    return remote, local


def _make_default_local_path(remote_path):
    """Generates a default local cache path based on a remote path.

    The default local path is created under `os.getenv("TMP_DIR", "/tmp")`,
    mirroring the directory structure of the `remote_path` (excluding the first component,
    typically the bucket name or scheme).

    Args:
        remote_path (str): The remote path.

    Returns:
        str: The generated default local path.
    """
    return str(Path(*[os.getenv("TMP_DIR", "/tmp")] + list(Path(remote_path).parts[1:])))


def _make_weighting_sequences(remote, proportion=None, repeat=None, choose=None):
    """Ensures weighting sequences (proportion, repeat, choose) match the length of remote paths.

    If a weighting sequence is provided, it must have the same length as the `remote` sequence.
    If a weighting sequence is None, it's initialized as a list of Nones with the same length
    as `remote`.

    Args:
        remote (Sequence[str]): Sequence of remote paths.
        proportion (list, optional): Stream proportions. Defaults to None.
        repeat (list, optional): Stream repeats. Defaults to None.
        choose (list, optional): Stream chooses. Defaults to None.

    Returns:
        tuple[list, list, list]: A tuple containing the validated or initialized
            proportion, repeat, and choose lists.

    Raises:
        ValueError: If any provided weighting sequence has a different length than `remote`.
    """
    weights = {"proportion": proportion, "repeat": repeat, "choose": choose}
    for name, weight in weights.items():
        if weight is not None and len(remote) != len(weight):
            ValueError(f"{name} must be the same length as remote, got lengths {len(remote)} and {len(weight)}")
    proportion = weights["proportion"] if weights["proportion"] is not None else [None] * len(remote)
    repeat = weights["repeat"] if weights["repeat"] is not None else [None] * len(remote)
    choose = weights["choose"] if weights["choose"] is not None else [None] * len(remote)
    return proportion, repeat, choose