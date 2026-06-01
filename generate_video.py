# SPDX-License-Identifier: LicenseRef-LMU-CompVis-NC-Research-1.0
# SPDX-FileCopyrightText: Copyright 2026 Ulrich Prestel, Stefan Baumann et al., CompVis @ LMU Munich

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import torch
from torchvision import transforms as T
from jaxtyping import Float
from PIL import Image
import imageio.v2 as imageio

import rayder.model
from rayder.model import RayDer, Camera


def load_images(image_dir: Path, resolution: int) -> Float[torch.Tensor, "n h w 3"]:
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted((p for p in image_dir.iterdir() if p.suffix.lower() in exts))
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    transform = T.Compose(
        [
            T.Lambda(lambda img: T.functional.center_crop(img, min(img.size))),
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(),
        ]
    )
    imgs = [transform(Image.open(p).convert("RGB")) for p in paths]
    return (torch.stack(imgs) * 2 - 1).permute(0, 2, 3, 1)


def load_checkpoint(path: Path, preset: Literal["XS", "S", "B", "L"], device: torch.device) -> RayDer:
    model = getattr(rayder.model, f"RayDer_{preset.upper()}")()
    sd = torch.load(path, map_location="cpu", weights_only=True)
    # TODO: remove
    # Convert from old state dict format to new one
    if "camera_tokens_2.weight" in sd:
        sd["nvs_tokens.weight"] = sd.pop("camera_tokens_2.weight")
    model.load_state_dict(sd, strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)


@torch.no_grad()
def generate_video(
    model: RayDer,
    images: Float[torch.Tensor, "n h w 3"],
    steps_per_pair: int = 10,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> Float[torch.Tensor, "t h w 3"]:
    x = images[None].to(device)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
        cams = model.predict_cameras(x=x)

    N = images.shape[0]
    all_frames: list[Float[torch.Tensor, "s h w 3"]] = []

    # Batch over pairs of views to reduce VRAM usage - cna be all just done in a single forward if intended
    for i in range(N - 1):
        cams_interp = Camera.interpolate(
            cams[:, i : i + 1], cams[:, i + 1 : i + 2], torch.linspace(0, 1, steps_per_pair, device=device)[None]
        )

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            views = model.predict_views(x_in=x, cam_in=cams, cam_target=cams_interp, state_target=None)

        all_frames.append(views[0])

    return torch.cat(all_frames, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="RayDer: generate interpolated novel-view video")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output.mp4"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps_per_pair", type=int, default=10)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--preset", choices=["S", "B", "L"], default="L")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model ({args.preset}) from {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, args.preset, device)

    images = load_images(args.image_dir, args.resolution)
    print(f"Loaded {images.shape[0]} images from {args.image_dir}  (resolution={args.resolution})")

    print(f"Generating video ({args.steps_per_pair} steps/pair, {images.shape[0]} input views)")
    frames = generate_video(model, images, args.steps_per_pair, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        str(args.output), ((frames.float().clamp(-1, 1) / 2 + 0.5) * 255).byte().cpu().numpy(), fps=args.fps
    )
    print(f"Saved {frames.shape[0]} frames -> {args.output}")


if __name__ == "__main__":
    main()
