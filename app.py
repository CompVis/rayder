# SPDX-License-Identifier: LicenseRef-LMU-CompVis-NC-Research-1.0
# SPDX-FileCopyrightText: Copyright 2026 Ulrich Prestel, Stefan Baumann et al., CompVis @ LMU Munich

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Sequence

import gradio as gr
import imageio.v2 as imageio
import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torchvision import transforms as T

from rayder.model import RayDer, RayDer_L, Camera

RESOLUTION = 576
FPS = 15
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

_model: RayDer | None = None
_device: torch.device = torch.device("cpu")
_ckpt_override: Path | None = None

_transform = T.Compose(
    [
        T.Lambda(lambda img: T.functional.center_crop(img, min(img.size))),
        T.Resize((RESOLUTION, RESOLUTION), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
    ]
)

_examples: list[list[str]] = []


def _load_model() -> RayDer:
    global _model
    if _model is not None:
        return _model
    if _ckpt_override:
        model = RayDer_L()
        sd = torch.load(_ckpt_override, map_location="cpu", weights_only=True)
        # TODO: remove
        if "camera_tokens_2.weight" in sd:
            sd["nvs_tokens.weight"] = sd.pop("camera_tokens_2.weight")
        model.load_state_dict(sd, strict=True)
        _model = model.eval().requires_grad_(False).to(_device)
    else:
        _model = torch.hub.load("CompVis/rayder", "rayder_l_576").to(_device)
    return _model


@torch.no_grad()
def _generate_video(image_paths: Sequence[str], steps_per_pair: int = 10) -> tuple[list[np.ndarray], str]:
    model = _load_model()
    x = torch.stack([_transform(Image.open(p).convert("RGB")) for p in sorted(image_paths, key=lambda p: Path(p).name)])
    x = x.mul(2).sub(1).permute(0, 2, 3, 1)[None].to(_device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=_device.type == "cuda"):
        cams = model.predict_cameras(x=x)

    all_frames: list[Float[torch.Tensor, "s h w 3"]] = []
    for i in range(x.shape[1] - 1):
        alpha = torch.linspace(0, 1, steps_per_pair, device=_device)[None]
        cams_interp = Camera.interpolate(cams[:, i : i + 1], cams[:, i + 1 : i + 2], alpha)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=_device.type == "cuda"):
            views = model.predict_views(x_in=x, cam_in=cams, cam_target=cams_interp, state_target=None)
        all_frames.append(views[0])

    frames_uint8 = ((torch.cat(all_frames).float().clamp(-1, 1) / 2 + 0.5) * 255).byte().cpu().numpy()
    video_path = Path(tempfile.mkdtemp()) / "video.mp4"
    imageio.mimsave(str(video_path), frames_uint8, fps=FPS)
    return list(frames_uint8), str(video_path)


def _discover_examples() -> list[list[str]]:
    if not EXAMPLES_DIR.is_dir():
        return []
    results: list[list[str]] = []
    for folder in sorted(EXAMPLES_DIR.iterdir()):
        if not folder.is_dir():
            continue
        files = sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if files:
            results.append(files)
    return results


def _to_gallery(paths: Sequence[str]) -> list[np.ndarray]:
    return [np.array(Image.open(p).convert("RGB")) for p in paths]


def _on_upload(files: Sequence[object]):
    paths = [f.name if hasattr(f, "name") else str(f) for f in (files or [])]
    if not paths:
        raise gr.Error("Please upload images.")
    return _to_gallery(paths), paths


def _on_example_select(evt: gr.SelectData):
    paths = _examples[evt.index]
    return paths, _to_gallery(paths), paths


def _run(paths: list[str] | None, steps_per_pair: int, progress=gr.Progress()):
    if not paths:
        raise gr.Error("Please upload or select images first.")
    progress(0.1, desc="Running inference")
    gallery, video_path = _generate_video(paths, steps_per_pair)
    progress(1.0, desc="Done")
    return gallery, video_path


_TITLE = "RayDer: Scalable Self-Supervised Novel View Synthesis from Real-World Video"
_DESCRIPTION = """\
<div>
<a style="display:inline-block" href="https://compvis.github.io/rayder/"><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a style="display:inline-block; margin-left: .5em" href="https://github.com/CompVis/rayder"><img src='https://img.shields.io/github/stars/CompVis/rayder?style=social'></a>
</div>
Upload a set of views of a scene. RayDer estimates cameras, interpolates between them, and synthesizes novel views along the trajectory.
"""


def build_demo() -> gr.Blocks:
    global _examples
    _examples = _discover_examples()

    with gr.Blocks(title=_TITLE, theme=gr.themes.Ocean()) as demo:
        gr.Markdown(f"# {_TITLE}")
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=2):
                image_block = gr.Files(label="Upload views", file_count="multiple", file_types=["image"])

                if _examples:
                    gr.Markdown("Or pick a curated example below.")
                    examples_gallery = gr.Gallery(value=[ex[0] for ex in _examples][:5], label="Examples", columns=4)

                batch_state = gr.State()
                preprocessed = gr.Gallery(label="Input views", columns=4, height=256)
                steps_slider = gr.Slider(minimum=2, maximum=30, value=10, step=1, label="Steps per pair")
                run_btn = gr.Button("Generate Video", variant="primary")

            with gr.Column(scale=4):
                output_gallery = gr.Gallery(label="Predicted views", columns=4, height=256)
                render_video = gr.Video(label="Interpolation video", autoplay=True, height=400)

        if _examples:
            examples_gallery.select(
                fn=_on_example_select, inputs=None, outputs=[image_block, preprocessed, batch_state]
            )

        image_block.upload(fn=_on_upload, inputs=[image_block], outputs=[preprocessed, batch_state])
        run_btn.click(fn=_run, inputs=[batch_state, steps_slider], outputs=[output_gallery, render_video])

    return demo


def main() -> None:
    global _device, _ckpt_override
    parser = argparse.ArgumentParser(description="RayDer Gradio demo")
    parser.add_argument("--ckpt", type=Path, default=None, help="Local checkpoint path (uses torch.hub if omitted)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    _device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    _ckpt_override = args.ckpt

    demo = build_demo()
    demo.queue().launch(share=args.share, server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
