# SPDX-License-Identifier: LicenseRef-LMU-CompVis-NC-Research-1.0
# SPDX-FileCopyrightText: Copyright 2026 Ulrich Prestel, Stefan Baumann et al., CompVis @ LMU Munich

from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from pydoc import locate
from typing import Any, Callable, Dict, Iterator, Literal, List
import io
import math
import random

import numpy as np
import torch
import torchvision.transforms.v2 as TVT
import torchvision.transforms.v2.functional as TF
from jaxtyping import Float, UInt8
import webdataset as wds
from webdataset.filters import pipelinefilter, reraise_exception

import av


def _map_many(data, f, handler=reraise_exception):
    """WebDataset map stage that can yield multiple samples per input sample."""
    for sample in data:
        try:
            results = f(sample)
        except Exception as exn:
            if handler(exn):
                continue
            break
        for i, r in enumerate(results):
            if r is None:
                continue
            if isinstance(sample, dict) and isinstance(r, dict):
                r["__key__"] = f"{sample.get('__key__', '')}-{i}"
            yield r


map_many = pipelinefilter(_map_many)


def dict_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack tensors; non-tensors are left untouched."""
    collated: Dict[str, Any] = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values
    return collated


class AdditionalDataDecoder(ABC):
    @abstractmethod
    def decode(self, sample: dict[str, Any]) -> dict[str, Any]:
        pass


def _decode_video_av(
    data: bytes, fps: float | None, clip_length: int, frame_index_jitter: bool | Literal["fully_random"]
) -> tuple[dict[int, UInt8[np.ndarray, "h w c"]], list[list[int]]]:
    with io.BytesIO(data) as buf, av.open(buf) as container:
        if not container.streams.video:
            raise Exception("No video stream found.")
        stream = container.streams.video[0]
        n_frames = stream.frames if stream.frames > 0 else 1000
        native_fps = float(stream.average_rate) if stream.average_rate else None
        if not native_fps:
            raise Exception("No native FPS found.")

        if fps is not None:
            if native_fps < fps:
                raise Exception(f"Native FPS {native_fps} is less than requested {fps}")
            step = int(round(native_fps / fps)) if fps else 1
        else:
            step = 1
        n_steps = n_frames // step
        num_chunks = n_steps // clip_length
        if num_chunks < 1:
            raise Exception(
                f"Insufficient video length (expected {clip_length} frames at {fps} FPS, "
                f"got {n_frames} frames at {native_fps} FPS)"
            )

        i_frame_chunk_start = [random.randrange(0, n_frames - clip_length * step + 1) for _ in range(num_chunks)]
        i_frames_chunks = [[i_start + j * step for j in range(clip_length)] for i_start in i_frame_chunk_start]
        if frame_index_jitter == "fully_random":
            i_min = max(0, min(n_frames - 1, min(i_frame_chunk_start) - step // 2 + 1))
            i_max = min(n_frames - 1, max(i_frame_chunk_start) + step // 2)
            i_frames_chunks = [[random.randint(i_min, i_max) for _ in range(clip_length)] for _ in range(num_chunks)]
        elif frame_index_jitter:
            i_frames_chunks = [
                [max(0, min(n_frames - 1, i + random.randint(-step // 2 + 1, step // 2))) for i in chunk]
                for chunk in i_frames_chunks
            ]
        i_frames = {i for chunk in i_frames_chunks for i in chunk}

        frames = {}
        i_frame = 0
        done = False
        for packet in container.demux(video=0):  # type: ignore[attr-defined]
            for frame in packet.decode():
                if i_frame in i_frames:
                    frames[i_frame] = frame.to_ndarray(format="rgb24")  # type: ignore[attr-defined]
                i_frame += 1
                done = len(frames) == len(i_frames)
                if done:
                    break
            if done:
                break
        container.close()

        if len(frames) != len(i_frames):
            raise Exception(f"Decoded {len(frames)} frames, expected {len(i_frames)}")

        return frames, i_frames_chunks


class SmolVideoLoader:
    def __init__(
        self,
        data_paths: List[str] | str,
        clip_length: int,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        fps: int | None = None,
        frame_index_jitter: (
            bool | Literal["fully_random"]
        ) = False,
        batch_size: int = 1,
        shuffle: int = 0,
        num_workers: int = 0,
        additional_data_decoders: list[str] | None = None,
        video_backend: Literal["av", "decord", "video_reader_rs"] = "av",
        extra_aug_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.tars = [
            str(f)
            for d in ([data_paths] if isinstance(data_paths, str) else data_paths)
            for f in Path(d).glob("**/*.tar")
        ]
        assert len(self.tars) > 0
        if video_backend != "av":
            raise ValueError("RayDer training data uses the Flow Poke standard backend combo: WebDataset + PyAV.")

        self.dataset = wds.DataPipeline(
            wds.ResampledShards(urls=self.tars),
            wds.detshuffle(),
            wds.split_by_node,
            wds.split_by_worker,
            partial(wds.tarfile_samples, handler=wds.warn_and_continue),
            *([wds.shuffle(shuffle)] if shuffle != 0 else []),
            map_many(self._decode),
            *([wds.shuffle(10)] if shuffle >= 10 else []),
            wds.select(lambda d: d.get("valid", True)),
            wds.batched(batch_size, partial=False, collation_fn=dict_collate_fn),
        )
        self.clip_length = clip_length
        self.fps = fps
        self.frame_index_jitter = frame_index_jitter
        self.transform = transform
        self.extra_aug_transform = extra_aug_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.additional_data_decoders: list[AdditionalDataDecoder] = [
            locate(d)() for d in (additional_data_decoders or [])
        ]
        self.video_backend = video_backend

    def _decode(self, sample: dict[str, Any]) -> Iterator[dict[str, Any]]:
        try:
            video_key = next((ext for ext in ("video.mpg", "mp4", "avi", "webm") if ext in sample), None)
            if not video_key:
                raise Exception(f"No video found in sample. Available keys: {list(sample.keys())}")

            video_data = sample.pop(video_key)
            frames, i_frames_chunks = _decode_video_av(
                video_data,
                self.fps,
                self.clip_length,
                self.frame_index_jitter,
            )

            frames_chunks: list[Float[torch.Tensor, "t c h w"]] = [
                torch.from_numpy(np.stack([frames[i] for i in chunk]).astype(np.float32) / 127.5 - 1).permute(
                    0, 3, 1, 2
                )
                for chunk in i_frames_chunks
            ]
            if self.transform is not None:
                frames_chunks = [self.transform(frames) for frames in frames_chunks]

            if self.extra_aug_transform is not None:
                frames_chunks_2 = [self.extra_aug_transform(frames) for frames in frames_chunks]
                for frames, frames_2, i_f in zip(frames_chunks, frames_chunks_2, i_frames_chunks):
                    d = sample | {"x": frames, "x_extra_aug": frames_2, "i_f": i_f}
                    for decoder in self.additional_data_decoders:
                        d = d | decoder.decode(d)
                    yield d
            else:
                for frames, i_f in zip(frames_chunks, i_frames_chunks):
                    d = sample | {"x": frames, "i_f": i_f}
                    for decoder in self.additional_data_decoders:
                        d = d | decoder.decode(d)
                    yield d
        except Exception as e:
            print(f"Error decoding sample {sample.get('__key__', 'unknown')}: {e}", flush=True)
            return {"valid": False}

    def __iter__(self):
        yield from wds.WebLoader(
            self.dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def _transform(frames: torch.Tensor, size: int) -> torch.Tensor:
    H, W = frames.shape[-2], frames.shape[-1]
    L = min(H, W)
    starth = H // 2 - (L // 2)
    startw = W // 2 - (L // 2)
    frames = frames[..., starth : (starth + L), startw : (startw + L)]
    return TVT.functional.resize(frames, [size, size], interpolation=TVT.InterpolationMode.BILINEAR, antialias=True)


def get_minimal_video_transform(size: int) -> Callable[[torch.Tensor], torch.Tensor]:
    return partial(_transform, size=size)

