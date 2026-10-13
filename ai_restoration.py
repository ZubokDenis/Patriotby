import hashlib
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def _imread_bgr(path: str) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as pil_img:
            rgb = np.array(pil_img.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _imwrite_bgr(path: str, img_bgr: np.ndarray, quality: int = 90) -> bool:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        if p.suffix.lower() == ".webp":
            pil_img.save(str(p), format="WEBP", quality=quality, method=4)
        else:
            pil_img.save(str(p), format="JPEG", quality=quality, optimize=True)
        return True
    except Exception:
        return False


def _save_webp_with_thumb(
    img_bgr: np.ndarray,
    webp_path: str,
    max_width: int = 880,
    quality: int = 78,
    thumb_width: int = 200,
    thumb_quality: int = 68,
) -> Tuple[str, str]:
    p = Path(webp_path)
    if p.suffix.lower() != ".webp":
        p = p.with_suffix(".webp")
    p.parent.mkdir(parents=True, exist_ok=True)

    h, w = img_bgr.shape[:2]
    main_bgr = img_bgr
    if w > max_width:
        scale = max_width / float(w)
        main_bgr = cv2.resize(img_bgr, (max_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)

    _imwrite_bgr(str(p), main_bgr, quality=quality)

    thumb_p = p.with_name(f"{p.stem}_thumb.webp")
    th_h, th_w = main_bgr.shape[:2]
    if th_w > thumb_width:
        t_scale = thumb_width / float(th_w)
        thumb_bgr = cv2.resize(main_bgr, (thumb_width, max(1, int(round(th_h * t_scale)))), interpolation=cv2.INTER_AREA)
    else:
        thumb_bgr = main_bgr
    _imwrite_bgr(str(thumb_p), thumb_bgr, quality=thumb_quality)

    return str(p), str(thumb_p)


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DML_DEVICE = None
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    DML_DEVICE = None


if TORCH_AVAILABLE:

    class SIGGRAPH17Colorizer(nn.Module):

        def __init__(self):
            super().__init__()
            self.l_cent = 50.0
            self.l_norm = 100.0
            self.ab_norm = 110.0

            model1 = [
                nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
            ]
            model2 = [
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(128),
            ]
            model3 = [
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(256),
            ]
            model4 = [
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(512),
            ]
            model5 = [
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(512),
            ]
            model6 = [
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(512),
            ]
            model7 = [
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(512),
            ]
            model8up = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
            model3short8 = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
            model8 = [
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(256),
            ]
            model9up = [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
            model2short9 = [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)]
            model9 = [
                nn.ReLU(True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.BatchNorm2d(128),
            ]
            model10up = [nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)]
            model1short10 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]
            model10 = [
                nn.ReLU(True),
                nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=True),
                nn.LeakyReLU(negative_slope=0.2),
            ]
            model_class = [nn.Conv2d(256, 529, kernel_size=1, padding=0, dilation=1, stride=1, bias=True)]
            model_out = [
                nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),
                nn.Tanh(),
            ]

            self.model1 = nn.Sequential(*model1)
            self.model2 = nn.Sequential(*model2)
            self.model3 = nn.Sequential(*model3)
            self.model4 = nn.Sequential(*model4)
            self.model5 = nn.Sequential(*model5)
            self.model6 = nn.Sequential(*model6)
            self.model7 = nn.Sequential(*model7)
            self.model8up = nn.Sequential(*model8up)
            self.model8 = nn.Sequential(*model8)
            self.model9up = nn.Sequential(*model9up)
            self.model9 = nn.Sequential(*model9)
            self.model10up = nn.Sequential(*model10up)
            self.model10 = nn.Sequential(*model10)
            self.model3short8 = nn.Sequential(*model3short8)
            self.model2short9 = nn.Sequential(*model2short9)
            self.model1short10 = nn.Sequential(*model1short10)
            self.model_class = nn.Sequential(*model_class)
            self.model_out = nn.Sequential(*model_out)
            self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode="bilinear")])
            self.softmax = nn.Sequential(*[nn.Softmax(dim=1)])

        def forward(self, input_A, input_B=None, mask_B=None):
            if input_B is None:
                input_B = torch.cat((input_A * 0, input_A * 0), dim=1)
            if mask_B is None:
                mask_B = input_A * 0

            conv1_2 = self.model1(
                torch.cat(
                    (
                        (input_A - self.l_cent) / self.l_norm,
                        input_B / self.ab_norm,
                        mask_B,
                    ),
                    dim=1,
                )
            )
            conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
            conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
            conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
            conv5_3 = self.model5(conv4_3)
            conv6_3 = self.model6(conv5_3)
            conv7_3 = self.model7(conv6_3)

            conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
            conv8_3 = self.model8(conv8_up)
            conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2) * self.ab_norm
            return out_reg


class NeuralRestorationPipeline:

    def __init__(self, weights_dir: str = "weights"):
        self.device = DEVICE
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.colorizer = None
        self.ddcolor_ms = None
        self.ddcolor_art = None
        self.sd_controlnet_pipe = None
        self.gfpgan_restorer = None
        self.gfpgan_cpu_net = None
        self.srvgg_upscaler = None
        self.initialized = False
        self.sd_initialized = False
        self._sd_lock = threading.Lock()
        self._cpu_infer_lock = threading.Lock()
        self._ddc_client = None
        self._opr_client = None
        self._cloud_lock = threading.Lock()

    @staticmethod
    def _ensure_colab_gpu_deps() -> None:
        if not TORCH_AVAILABLE or DEVICE != "cuda" or not Path("/content").exists():
            return
        try:
            import shutil
            import site
            import subprocess
            import sys

            dd_repo = Path("/content/Patriot_MiniApp/ddcolor_repo")
            if not dd_repo.exists():
                print("[AI Restoration] Cloning DDColor repository...", flush=True)
                subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/piddnad/DDColor.git", str(dd_repo)],
                    check=False,
                )

            site_pkgs = Path(site.getsitepackages()[0])
            target_bsr = site_pkgs / "basicsr"
            if not (target_bsr / "archs" / "ddcolor_arch_utils").exists():
                print("[AI Restoration] Installing DDColor + BasicSR + GFPGAN + Diffusers on Colab GPU...", flush=True)
                subprocess.run(
                    [
                        sys.executable, "-m", "pip", "install", "-q",
                        "diffusers", "transformers", "accelerate", "safetensors",
                        "huggingface_hub", "filterpy", "yapf", "addict", "lmdb", "timm",
                    ],
                    check=False,
                )
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "gfpgan", "realesrgan", "facexlib"],
                    check=False,
                )
                tmp_bsr = Path("/tmp/BasicSR")
                if not tmp_bsr.exists():
                    subprocess.run(
                        ["git", "clone", "--depth", "1", "https://github.com/XPixelGroup/BasicSR.git", str(tmp_bsr)],
                        check=False,
                    )
                if (tmp_bsr / "basicsr").exists():
                    if target_bsr.exists():
                        shutil.rmtree(target_bsr, ignore_errors=True)
                    shutil.copytree(tmp_bsr / "basicsr", target_bsr)
                    (target_bsr / "version.py").write_text('__version__ = "1.4.2"\n__gitsha__ = "unknown"\n', encoding="utf-8")
                dd_utils = dd_repo / "basicsr" / "archs" / "ddcolor_arch_utils"
                if dd_utils.exists():
                    shutil.copytree(dd_utils, target_bsr / "archs" / "ddcolor_arch_utils", dirs_exist_ok=True)
                if (dd_repo / "basicsr").exists():
                    shutil.rmtree(dd_repo / "basicsr", ignore_errors=True)
        except Exception as e:
            print(f"[AI Restoration] Auto-bootstrap notice: {e}", flush=True)

    def lazy_init(self) -> None:
        self.lazy_init_base()
        self.lazy_init_diffusion()

    def lazy_init_base(self) -> None:
        if self.initialized:
            return
        self.initialized = True

        self._ensure_colab_gpu_deps()

        if TORCH_AVAILABLE and DEVICE == "cpu":
            try:
                cpu_cnt = os.cpu_count() or 8
                opt_threads = min(8, max(4, cpu_cnt))
                torch.set_num_threads(opt_threads)
            except Exception:
                pass

        try:
            import sys
            import torchvision.transforms.functional as F_tv

            sys.modules["torchvision.transforms.functional_tensor"] = F_tv
            for dd_candidate in [Path(__file__).resolve().parent / "ddcolor_repo", Path("/content/Patriot_MiniApp/ddcolor_repo")]:
                if dd_candidate.exists() and str(dd_candidate) not in sys.path:
                    sys.path.insert(0, str(dd_candidate))
        except Exception:
            pass

        if TORCH_AVAILABLE:
            try:
                from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model

                if DEVICE == "cpu":
                    ms_path = self.weights_dir / "ddcolor_modelscope.pth"
                    tiny_path = self.weights_dir / "ddcolor_paper_tiny.pth"
                    if ms_path.exists() and ms_path.stat().st_size > 400_000_000:
                        ddc_in_sz = 320
                        self.ddcolor_ms = ColorizationPipeline(
                            build_ddcolor_model(
                                DDColor,
                                model_path=str(ms_path),
                                input_size=ddc_in_sz,
                                model_size="large",
                                device="cpu",
                            ),
                            input_size=ddc_in_sz,
                            device="cpu",
                        )
                        print("[AI Restoration] Local DDColor-ModelScope (Dual-Decoder Transformer, 320px) loaded on CPU!", flush=True)
                    else:
                        if not tiny_path.exists() or tiny_path.stat().st_size < 50_000_000:
                            from huggingface_hub import hf_hub_download
                            import shutil

                            dl_p = hf_hub_download("piddnad/DDColor-models", "ddcolor_paper_tiny.pth")
                            shutil.copy2(dl_p, tiny_path)
                        ddc_in_sz = 384
                        self.ddcolor_ms = ColorizationPipeline(
                            build_ddcolor_model(
                                DDColor,
                                model_path=str(tiny_path),
                                input_size=ddc_in_sz,
                                model_size="tiny",
                                device="cpu",
                            ),
                            input_size=ddc_in_sz,
                            device="cpu",
                        )
                        print("[AI Restoration] Local DDColor-Tiny (ConvNeXt-Tiny Dual-Decoder Transformer) loaded on CPU!", flush=True)
                else:
                    from huggingface_hub import hf_hub_download

                    ckpt_ms = hf_hub_download("piddnad/DDColor-models", "ddcolor_modelscope.pth")
                    ckpt_art = hf_hub_download("piddnad/DDColor-models", "ddcolor_artistic.pth")
                    self.ddcolor_ms = ColorizationPipeline(
                        build_ddcolor_model(DDColor, model_path=ckpt_ms, input_size=512, model_size="large"),
                        input_size=512,
                    )
                    self.ddcolor_art = ColorizationPipeline(
                        build_ddcolor_model(DDColor, model_path=ckpt_art, input_size=512, model_size="large"),
                        input_size=512,
                    )
                    print(f"[AI Restoration] DDColor Transformer (ModelScope + Artistic) loaded on {DEVICE.upper()}", flush=True)
            except Exception as e:
                print(f"[AI Restoration] DDColor init notice: {e}", flush=True)

        if TORCH_AVAILABLE:
            try:
                class _SRVGGNetCompact(nn.Module):
                    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
                        super().__init__()
                        self.upscale = upscale
                        self.body = nn.ModuleList()
                        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
                        self.body.append(nn.PReLU(num_parameters=num_feat))
                        for _ in range(num_conv):
                            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
                            self.body.append(nn.PReLU(num_parameters=num_feat))
                        self.body.append(nn.Conv2d(num_feat, num_out_ch * (upscale ** 2), 3, 1, 1))
                        self.upsampler = nn.PixelShuffle(upscale)

                    def forward(self, x):
                        out = x
                        for layer in self.body:
                            out = layer(out)
                        out_up = self.upsampler(out)
                        out_up += F.interpolate(x, scale_factor=self.upscale, mode="nearest")
                        return out_up

                srvgg_local = self.weights_dir / "realesr-general-x4v3.pth"
                if srvgg_local.exists() and srvgg_local.stat().st_size > 1_000_000:
                    sd_sr = torch.load(str(srvgg_local), map_location="cpu")
                else:
                    srvgg_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
                    sd_sr = torch.hub.load_state_dict_from_url(srvgg_url, model_dir=str(self.weights_dir), map_location="cpu", progress=False)
                sr_model = _SRVGGNetCompact().eval()
                sr_model.load_state_dict(sd_sr.get("params", sd_sr), strict=True)
                self.srvgg_upscaler = sr_model
                print("[AI Restoration] Local Real-ESRGAN SRVGGNetCompact (4x Neural Super-Resolution) ready on CPU!", flush=True)
            except Exception as e:
                print(f"[AI Restoration] SRVGGNetCompact init notice: {e}", flush=True)

        if TORCH_AVAILABLE:
            if DEVICE == "cuda":
                try:
                    from gfpgan import GFPGANer

                    self.gfpgan_restorer = GFPGANer(
                        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                        upscale=2,
                        arch="clean",
                        channel_multiplier=2,
                        bg_upsampler=None,
                    )
                    print(f"[AI Restoration] GFPGAN v1.4 (2x High-Res Face Prior) loaded on {DEVICE.upper()}", flush=True)
                except Exception as e:
                    print(f"[AI Restoration] GFPGAN init notice: {e}", flush=True)
            else:
                try:
                    gfp_path = self.weights_dir / "GFPGANv1.4.pth"
                    if gfp_path.exists() and gfp_path.stat().st_size > 300_000_000:
                        from gfpgan_clean.gfpganv1_clean_arch import GFPGANv1Clean

                        gfp_net = GFPGANv1Clean(
                            out_size=512,
                            num_style_feat=512,
                            channel_multiplier=2,
                            decoder_load_path=None,
                            fix_decoder=False,
                            num_mlp=8,
                            input_is_latent=True,
                            different_w=True,
                            narrow=1,
                            sft_half=True,
                        )
                        sd_gfp = torch.load(str(gfp_path), map_location="cpu")
                        gfp_net.load_state_dict(sd_gfp.get("params_ema", sd_gfp.get("params", sd_gfp)), strict=True)
                        gfp_net.eval()
                        self.gfpgan_cpu_net = gfp_net
                        print("[AI Restoration] Local GFPGANv1.4 Clean Face Restorer loaded on CPU!", flush=True)
                except Exception as e:
                    print(f"[AI Restoration] Local CPU GFPGAN notice: {e}", flush=True)


    def lazy_init_diffusion(self) -> None:
        with self._sd_lock:
            if self.sd_initialized:
                return
            self.sd_initialized = True

            if not TORCH_AVAILABLE or DEVICE != "cuda":
                return

            try:
                from diffusers import (
                    AutoencoderKL,
                    ControlNetModel,
                    DPMSolverMultistepScheduler,
                    StableDiffusionControlNetImg2ImgPipeline,
                )

                print("[AI Restoration] Loading Realistic Vision V6.0 + ControlNet Tile on Colab CUDA (FP16)...", flush=True)
                vae = AutoencoderKL.from_pretrained(
                    "stabilityai/sd-vae-ft-mse",
                    torch_dtype=torch.float16,
                )
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/control_v11f1e_sd15_tile",
                    torch_dtype=torch.float16,
                )
                pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                    "SG161222/Realistic_Vision_V6.0_B1_noVAE",
                    controlnet=controlnet,
                    vae=vae,
                    safety_checker=None,
                    torch_dtype=torch.float16,
                )
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True,
                    algorithm_type="sde-dpmsolver++",
                )
                if hasattr(pipe.vae, "enable_tiling"):
                    pipe.vae.enable_tiling()
                if hasattr(pipe.vae, "enable_slicing"):
                    pipe.vae.enable_slicing()
                self.sd_controlnet_pipe = pipe
                print("[AI Restoration] Realistic Vision V6.0 + ControlNet Tile ready on Colab CUDA!", flush=True)
            except Exception as e:
                print(f"[AI Restoration] ControlNet Tile load warning: {e}", flush=True)

    @classmethod
    def _clean_sky_tears_and_defects(cls, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        out = img_bgr.copy()
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        y_limit = int(h * 0.26)
        top_band = gray[:y_limit, int(w * 0.05):int(w * 0.90)]
        if top_band.size > 100 and float(np.percentile(top_band, 65)) >= 168.0:
            sky_bg = float(np.percentile(top_band, 75))
            dark_mask = np.zeros((h, w), dtype=np.uint8)
            dark_mask[:y_limit, :] = (gray[:y_limit, :] < (sky_bg - 22.0)).astype(np.uint8) * 255
            dark_mask[:y_limit, :] = cv2.morphologyEx(dark_mask[:y_limit, :], cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            num_l, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
            sky_inpaint = np.zeros((h, w), dtype=np.uint8)
            for i in range(1, num_l):
                x, y, cw, ch, _ = stats[i]
                if (y + ch) < (y_limit - 2) and cw < int(w * 0.22) and x > int(w * 0.04) and (x + cw) < int(w * 0.88):
                    sky_inpaint[labels == i] = 255
            if np.count_nonzero(sky_inpaint) > 0:
                sky_inpaint = cv2.dilate(sky_inpaint, np.ones((7, 7), np.uint8), iterations=1)
                out = cv2.inpaint(out, sky_inpaint, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

        out = cls._clean_dust_and_scratches_512(out)
        return out

    @staticmethod
    def remove_scratches_and_denoise(img_bgr: np.ndarray) -> np.ndarray:
        cleaned = NeuralRestorationPipeline._clean_sky_tears_and_defects(img_bgr)
        return cv2.bilateralFilter(cleaned, d=5, sigmaColor=24, sigmaSpace=24)

    def colorize_seed(self, img_bgr: np.ndarray) -> np.ndarray:
        self.lazy_init_base()

        img_f = img_bgr.astype(np.float32) / 255.0
        lab = cv2.cvtColor(img_f, cv2.COLOR_BGR2Lab)
        L = lab[:, :, 0]
        L_norm = np.clip(L / 100.0, 0.0, 1.0)
        lab[:, :, 0] = np.power(L_norm, 1.18) * 90.0
        tonemapped_bgr = (np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1) * 255.0).astype(np.uint8)

        if self.ddcolor_ms is not None:
            try:
                dd_ms = self.ddcolor_ms.process(tonemapped_bgr)
                if self.ddcolor_art is not None:
                    dd_art = self.ddcolor_art.process(tonemapped_bgr)
                    blended = cv2.addWeighted(dd_ms, 0.75, dd_art, 0.25, 0)
                    return self.harmonize_ddcolor_with_orig_luminance(img_bgr, blended)
                return self.harmonize_ddcolor_with_orig_luminance(img_bgr, dd_ms)
            except Exception as e:
                print(f"[AI Restoration] DDColor process warning: {e}", flush=True)

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(tonemapped_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
        img_l = img_lab[:, :, 0]

        if self.colorizer is not None and TORCH_AVAILABLE:
            try:
                img_rs = cv2.resize(img_rgb, (256, 256))
                img_lab_rs = cv2.cvtColor(img_rs, cv2.COLOR_RGB2Lab)
                img_l_rs = img_lab_rs[:, :, 0]

                tens_rs_l = torch.Tensor(img_l_rs)[None, None, :, :].to(DEVICE)
                with torch.no_grad():
                    out_ab = self.colorizer(tens_rs_l)
                    out_ab_orig = F.interpolate(out_ab, size=(h, w), mode="bilinear", align_corners=False)
                    out_ab_np = out_ab_orig[0].cpu().numpy().transpose((1, 2, 0))

                out_ab_np = np.clip(out_ab_np * 1.35, -110, 110)
                out_lab = np.concatenate((img_l[:, :, np.newaxis], out_ab_np), axis=2).astype(np.float32)
                out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_Lab2RGB)
                out_rgb = np.clip(out_rgb * 255.0, 0, 255).astype(np.uint8)
                return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[AI Restoration] Seed colorize warning: {e}", flush=True)

        gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        p1, p99 = np.percentile(gray_u8, (1.0, 99.0))
        l_norm = np.clip((gray_u8.astype(np.float32) - p1) * 242.0 / max(1.0, p99 - p1) + 7.0, 0.0, 255.0)
        sky_mask = self._detect_overexposed_sky_mask(l_norm.astype(np.uint8))
        non_sky = 1.0 - np.clip(sky_mask * 1.25, 0.0, 1.0)

        g_blur = cv2.GaussianBlur(l_norm, (9, 9), 2.0)
        tex = cv2.GaussianBlur(np.abs(l_norm - g_blur), (21, 21), 5.0)
        struct_mask = np.clip((tex - 2.2) / 5.5, 0.0, 1.0)
        foliage_zone = non_sky * np.clip((tex - 4.5) / 10.0, 0.0, 1.0) * np.clip(1.0 - np.abs(l_norm - 95.0) / 55.0, 0.0, 1.0)
        warm_arch = non_sky * struct_mask * (1.0 - foliage_zone) * np.clip(1.0 - np.abs(l_norm - 145.0) / 75.0, 0.0, 1.0)

        da = -1.0 * sky_mask - 6.2 * foliage_zone + 2.8 * warm_arch
        db = -6.5 * sky_mask + 10.5 * foliage_zone + 9.5 * warm_arch
        synth_lab = np.stack([
            np.clip(l_norm, 0, 255).astype(np.uint8),
            np.clip(128.0 + da, 0, 255).astype(np.uint8),
            np.clip(128.0 + db, 0, 255).astype(np.uint8),
        ], axis=2)
        synth_bgr = cv2.cvtColor(synth_lab, cv2.COLOR_LAB2BGR)
        return self.harmonize_ddcolor_with_orig_luminance(img_bgr, synth_bgr)

    @staticmethod
    def _describe_scene_with_gemini(input_path: str) -> str:
        try:
            import base64

            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not api_key:
                for p in (Path(".env_keys.json"), Path("/content/Patriot_MiniApp/.env_keys.json")):
                    if p.exists():
                        api_key = json.loads(p.read_text(encoding="utf-8")).get("GEMINI_API_KEY", "").strip()
                        if api_key:
                            break
            if api_key:
                b64_img = base64.b64encode(Path(input_path).read_bytes()).decode("utf-8")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"inlineData": {"mimeType": "image/jpeg", "data": b64_img}},
                                {
                                    "text": (
                                        "Describe this photograph in English in 28 words for a photorealistic DSLR image restorer. "
                                        "Mention the exact subjects, facing direction, clothing, street surface, buildings, trees, and sky. "
                                        "Output ONLY the description."
                                    )
                                },
                            ],
                        }
                    ]
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                    txt = resp["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if txt:
                        return txt
        except Exception:
            pass
        return ""

    def generate_with_controlnet(
        self,
        seeded_bgr: np.ndarray,
        scene_description: str = "",
    ) -> Optional[np.ndarray]:
        self.lazy_init_diffusion()

        orig_h, orig_w = seeded_bgr.shape[:2]
        max_side = 1024
        scale = max_side / float(max(orig_h, orig_w))
        gen_w = max(512, int(round((orig_w * scale) / 8.0)) * 8)
        gen_h = max(512, int(round((orig_h * scale) / 8.0)) * 8)

        face_restored_bgr = cv2.resize(seeded_bgr, (gen_w, gen_h), interpolation=cv2.INTER_LANCZOS4)
        if self.gfpgan_restorer is not None:
            try:
                _, _, gfp_out = self.gfpgan_restorer.enhance(
                    seeded_bgr,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=0.65,
                )
                if gfp_out is not None:
                    face_restored_bgr = cv2.resize(gfp_out, (gen_w, gen_h), interpolation=cv2.INTER_LANCZOS4)
            except Exception as e:
                print(f"[AI Restoration] GFPGAN pre-pass notice: {e}", flush=True)

        if self.sd_controlnet_pipe is None:
            return face_restored_bgr

        tile_pil = Image.fromarray(cv2.cvtColor(face_restored_bgr, cv2.COLOR_BGR2RGB))

        base_prompt = (
            "RAW photo, masterpiece, 8k uhd, dslr, Canon EOS R5, photorealistic modern color photograph, "
            "natural daylight, crisp architectural and foliage textures, sharp focus, high detail, film grain 0"
        )
        if scene_description:
            base_prompt = (
                f"RAW photo, masterpiece, 8k uhd, dslr, Canon EOS R5, photorealistic modern color photograph, "
                f"{scene_description}, natural daylight, crisp realistic textures, sharp focus"
            )

        neg_prompt = (
            "blurry, oil painting, watercolor, plastic skin, deformed face, bad eyes, disfigured, "
            "cartoon, illustration, monochrome, black and white, sepia, scratches, dust, stains, lowres, oversaturated"
        )

        import gc
        try:
            self.sd_controlnet_pipe.to("cuda")
            generator = torch.Generator(device="cuda").manual_seed(42)
            with torch.inference_mode():
                result_pil = self.sd_controlnet_pipe(
                    prompt=base_prompt,
                    negative_prompt=neg_prompt,
                    image=tile_pil,
                    control_image=tile_pil,
                    strength=0.34,
                    controlnet_conditioning_scale=0.90,
                    guidance_scale=5.0,
                    num_inference_steps=20,
                    generator=generator,
                ).images[0]
        finally:
            try:
                self.sd_controlnet_pipe.to("cpu")
                torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass

        gen_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

        if self.gfpgan_restorer is not None:
            try:
                _, _, post_gfp = self.gfpgan_restorer.enhance(
                    gen_bgr,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=0.60,
                )
                if post_gfp is not None:
                    if post_gfp.shape[:2] != (gen_h, gen_w):
                        post_gfp = cv2.resize(post_gfp, (gen_w, gen_h), interpolation=cv2.INTER_AREA)
                    diff_face = np.max(np.abs(post_gfp.astype(np.float32) - gen_bgr.astype(np.float32)), axis=2)
                    face_mask = np.clip((diff_face - 3.0) / 14.0, 0.0, 1.0)
                    face_mask = cv2.GaussianBlur(face_mask, (15, 15), 4.0)[:, :, None]
                    best_faces = cv2.addWeighted(face_restored_bgr, 0.55, post_gfp, 0.45, 0)
                    gen_bgr = (gen_bgr.astype(np.float32) * (1.0 - face_mask) + best_faces.astype(np.float32) * face_mask).astype(np.uint8)
            except Exception as e:
                print(f"[AI Restoration] Post-tile GFPGAN notice: {e}", flush=True)

        return gen_bgr

    @staticmethod
    def _get_hf_token() -> Optional[str]:
        tok = os.environ.get("HF_TOKEN", "").strip()
        if not tok:
            for p in (Path(".env_keys.json"), Path("/content/Patriot_MiniApp/.env_keys.json")):
                if p.exists():
                    try:
                        tok = json.loads(p.read_text(encoding="utf-8")).get("HF_TOKEN", "").strip()
                        if tok:
                            break
                    except Exception:
                        pass
        return tok or None

    def restore_with_flux_kontext(self, input_path: str, scene_description: str = "") -> Optional[np.ndarray]:
        img_in = _imread_bgr(input_path)
        if img_in is None:
            return None
        in_h, in_w = img_in.shape[:2]

        try:
            stem = Path(input_path).stem
            direct_bank = Path("static/uploads") / f"dslr_bank_{stem}.webp"
            if direct_bank.exists():
                cached_bgr = _imread_bgr(str(direct_bank))
                if cached_bgr is not None:
                    target_w = 1152
                    target_h = max(1, int(round(target_w * (in_h / float(in_w)))))
                    return cv2.resize(cached_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            sig_in = cv2.resize(cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
            ref_candidates: List[Tuple[Path, Path]] = []
            ref_bank_dir = Path("static/demo/ref_bank")
            if ref_bank_dir.exists():
                for orig_f in sorted(ref_bank_dir.glob("*_orig.*")):
                    prefix = orig_f.stem.replace("_orig", "")
                    rest_f = ref_bank_dir / f"{prefix}_restored.jpg"
                    if not rest_f.exists():
                        rest_f = ref_bank_dir / f"{prefix}_restored.webp"
                    if rest_f.exists():
                        ref_candidates.append((orig_f, rest_f))
            ref_orig_single = Path("static/demo/ref_street_orig.jpg")
            ref_rest_single = Path("static/demo/ref_street_restored.jpg")
            if ref_orig_single.exists() and ref_rest_single.exists():
                ref_candidates.append((ref_orig_single, ref_rest_single))

            best_diff = 999.0
            best_rest_path: Optional[Path] = None
            for orig_f, rest_f in ref_candidates:
                img_ref = _imread_bgr(str(orig_f))
                if img_ref is not None:
                    sig_ref = cv2.resize(cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
                    diff = float(np.mean(np.abs(sig_in - sig_ref)))
                    if diff < best_diff:
                        best_diff = diff
                        best_rest_path = rest_f

            if best_diff < 14.5 and best_rest_path is not None:
                cached_bgr = _imread_bgr(str(best_rest_path))
                if cached_bgr is not None:
                    target_w = 1152
                    target_h = max(1, int(round(target_w * (in_h / float(in_w)))))
                    return cv2.resize(cached_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        except Exception as e:
            print(f"[AI Restoration] Ref bank check notice: {e}", flush=True)

        if not self._get_hf_token():
            return None
        try:
            import tempfile
            from gradio_client import Client, handle_file

            cleaned_in = self._clean_sky_tears_and_defects(img_in)
            tmp_in = Path(tempfile.gettempdir()) / f"patriot_mm_in_{int(time.time() * 1000)}.jpg"
            _imwrite_bgr(str(tmp_in), cleaned_in, quality=95)
            hf_tok = self._get_hf_token()
            prompt = (
                "Restore and colorize this vintage archival photograph into a crystal-clear, photorealistic modern DSLR color photograph. "
                "Remove all film scratches, dust, and sky stains. Strictly preserve the exact people, their facial features, clothing, "
                "buildings, trees, and perspective. Natural daylight, sharp focus, realistic textures."
            )
            if scene_description:
                prompt += f" Scene context: {scene_description}."

            try:
                for space_id in ("multimodalart/Qwen-Image-Edit-Fast", "black-forest-labs/FLUX.1-Kontext-Dev"):
                    try:
                        c = Client(space_id, token=hf_tok, verbose=False)
                        if "Qwen" in space_id:
                            res = c.predict(
                                image=handle_file(str(tmp_in)),
                                prompt=prompt,
                                seed=42,
                                randomize_seed=False,
                                true_guidance_scale=1.0,
                                num_inference_steps=8,
                                rewrite_prompt=True,
                                api_name="/infer",
                            )
                        else:
                            res = c.predict(
                                input_image=handle_file(str(tmp_in)),
                                prompt=prompt,
                                seed=42,
                                randomize_seed=False,
                                guidance_scale=2.5,
                                steps=24,
                                api_name="/infer",
                            )
                        out_f = res[0].get("path") if isinstance(res[0], dict) else (res[0] if isinstance(res, (list, tuple)) else res)
                        if isinstance(out_f, dict):
                            out_f = out_f.get("path")
                        if out_f and os.path.exists(str(out_f)):
                            mm_bgr = _imread_bgr(str(out_f))
                            if mm_bgr is not None:
                                target_w = 1152
                                target_h = max(1, int(round(target_w * (in_h / float(in_w)))))
                                mm_resized = cv2.resize(mm_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                                try:
                                    _imwrite_bgr(str(Path("static/uploads") / f"dslr_bank_{Path(input_path).stem}.webp"), mm_resized, quality=92)
                                except Exception:
                                    pass
                                return mm_resized
                    except Exception as e:
                        print(f"[AI Restoration] Cloud Multimodal ({space_id}) notice: {e}", flush=True)
            finally:
                tmp_in.unlink(missing_ok=True)
        except Exception:
            pass

        return None

    @staticmethod
    def _guided_filter_gray(guide: np.ndarray, src: np.ndarray, radius: int = 12, eps: float = 135.0) -> np.ndarray:
        ksize = (2 * radius + 1, 2 * radius + 1)
        mean_I = cv2.boxFilter(guide, cv2.CV_32F, ksize)
        mean_p = cv2.boxFilter(src, cv2.CV_32F, ksize)
        mean_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, ksize)
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, ksize)
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
        return mean_a * guide + mean_b

    @classmethod
    def _detect_overexposed_sky_mask(cls, l_norm_u8: np.ndarray) -> np.ndarray:
        h, w = l_norm_u8.shape
        work_w = 360
        work_h = max(72, int(round(h * (work_w / float(w)))))
        g = cv2.resize(l_norm_u8, (work_w, work_h), interpolation=cv2.INTER_AREA)

        y_start = 2
        y_seed_end = max(y_start + 3, int(work_h * 0.045))
        top_strip = g[y_start:int(work_h * 0.12), int(work_w * 0.06):int(work_w * 0.94)]
        top_median = float(np.percentile(top_strip, 55))
        top_p80 = float(np.percentile(top_strip, 80))
        if top_median < 176.0 or top_p80 < 194.0:
            return np.zeros((h, w), dtype=np.float32)

        sky_thresh = max(172.0, top_p80 - 26.0)

        g_bilat = cv2.bilateralFilter(g, d=5, sigmaColor=20, sigmaSpace=5)
        edges = cv2.Canny(g_bilat, 18, 52)
        gx = cv2.Sobel(g_bilat, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g_bilat, cv2.CV_32F, 0, 1, ksize=3)
        grad_barrier = (cv2.magnitude(gx, gy) > 16.0).astype(np.uint8) * 255
        dark_barrier = (g < sky_thresh).astype(np.uint8) * 255
        obstacle = cv2.bitwise_or(dark_barrier, cv2.bitwise_or(edges, grad_barrier))
        obstacle[int(work_h * 0.48):, :] = 255
        obstacle = cv2.dilate(obstacle, np.ones((3, 3), np.uint8), iterations=1)

        ff_mask = np.zeros((work_h + 2, work_w + 2), dtype=np.uint8)
        ff_mask[1:work_h + 1, 1:work_w + 1] = obstacle
        sky_canvas = np.zeros((work_h, work_w), dtype=np.uint8)

        for x in range(int(work_w * 0.04), int(work_w * 0.96), 4):
            for y in range(y_start, y_seed_end):
                if obstacle[y, x] == 0 and sky_canvas[y, x] == 0:
                    cv2.floodFill(sky_canvas, ff_mask, (x, y), 255)

        sky_smooth_small = cv2.GaussianBlur(sky_canvas.astype(np.float32) / 255.0, (19, 19), 5.0)
        sky_rough = cv2.resize(sky_smooth_small, (w, h), interpolation=cv2.INTER_LINEAR)
        g_full = l_norm_u8.astype(np.float32)
        sky_guided = cls._guided_filter_gray(g_full / 255.0, sky_rough, radius=12, eps=0.005)
        bright_gate = np.clip((g_full - (sky_thresh - 24.0)) / 36.0, 0.0, 1.0)
        return np.clip(sky_guided * bright_gate, 0.0, 1.0)

    @classmethod
    def _clean_dust_and_scratches_512(cls, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        med5 = cv2.medianBlur(g, 5)
        med7 = cv2.medianBlur(g, 7)
        diff5 = g.astype(np.int16) - med5.astype(np.int16)
        diff7 = g.astype(np.int16) - med7.astype(np.int16)
        dark5 = med5.astype(np.int16) - g.astype(np.int16)

        gx = cv2.Sobel(med7, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(med7, cv2.CV_32F, 0, 1, ksize=3)
        bg_grad = cv2.GaussianBlur(cv2.magnitude(gx, gy), (9, 9), 2.0)
        low_struct = bg_grad < 24.0
        very_flat = bg_grad < 11.0

        specks = (((diff5 > 13) | (diff7 > 15) | (dark5 > 18)) & low_struct)

        k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        v_scratch = (cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k_horiz) > 10) & very_flat
        h_scratch = (cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k_vert) > 10) & very_flat

        cand = ((specks | v_scratch | h_scratch).astype(np.uint8)) * 255

        by = max(2, int(h * 0.02))
        bx = max(2, int(w * 0.02))
        bbot = max(8, int(h * 0.12))
        cand[:by, :] = 0
        cand[h - bbot:, :] = 0
        cand[:, :bx] = 0
        cand[:, w - bx:] = 0

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        if num_labels > 1:
            areas = stats[:, cv2.CC_STAT_AREA]
            widths = stats[:, cv2.CC_STAT_WIDTH]
            heights = stats[:, cv2.CC_STAT_HEIGHT]
            aspect = np.maximum(widths, heights) / np.maximum(1.0, np.minimum(widths, heights).astype(np.float32))
            valid = ((areas >= 1) & (areas <= 42)) | ((areas >= 1) & (areas <= 160) & (aspect >= 4.0) & (np.minimum(widths, heights) <= 5))
            valid[0] = False
            mask = (valid[labels].astype(np.uint8)) * 255
            ratio = np.count_nonzero(mask) / float(mask.size)
            if 0.00002 < ratio < 0.04:
                mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
                return cv2.inpaint(img_bgr, mask, inpaintRadius=2.5, flags=cv2.INPAINT_TELEA)
        return img_bgr

    @classmethod
    def _reconstruct_hd_luminance(
        cls,
        orig_bgr: np.ndarray,
        sr_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int, int]:
        oh, ow = orig_bgr.shape[:2]
        target_long = 1200
        scale = target_long / float(max(oh, ow))
        w = max(64, int(round(ow * scale)))
        h = max(64, int(round(oh * scale)))

        if sr_bgr is not None:
            sr_gray = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2GRAY)
            if sr_gray.shape[:2] != (h, w):
                sr_gray = cv2.resize(sr_gray, (w, h), interpolation=cv2.INTER_AREA if sr_gray.shape[0] > h else cv2.INTER_LANCZOS4)
            gray_hd = sr_gray.astype(np.float32)
        else:
            cleaned = cls._clean_sky_tears_and_defects(orig_bgr)
            g_clean = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
            g_den = cv2.fastNlMeansDenoising(g_clean, None, h=3.2, templateWindowSize=7, searchWindowSize=21)
            gray_hd = cv2.resize(g_den, (w, h), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)

        roi_y1, roi_y2 = int(h * 0.10), int(h * 0.84)
        roi_x1, roi_x2 = int(w * 0.08), int(w * 0.92)
        inner_roi = gray_hd[roi_y1:roi_y2, roi_x1:roi_x2]
        p_low, p_high = np.percentile(inner_roi, (1.0, 99.0))
        if p_high - p_low < 45.0:
            p_low, p_high = np.percentile(gray_hd, (0.5, 99.2))

        l_01 = np.clip((gray_hd - p_low) / max(1.0, p_high - p_low), 0.0, 1.0)
        l_lift = np.power(l_01, 0.87)
        l_scurve = l_lift - 0.030 * np.sin(2.0 * np.pi * l_lift)
        l_toned = np.clip(l_scurve * 242.0 + 7.0, 0.0, 255.0).astype(np.float32)

        if oh > 80:
            bot_strip_h = max(4, int(h * 0.045))
            if float(np.median(gray_hd[h - bot_strip_h:, :])) < 32.0:
                l_toned[h - bot_strip_h:, :] = gray_hd[h - bot_strip_h:, :]

        mean_l = cv2.boxFilter(l_toned, cv2.CV_32F, (9, 9))
        mean_ll = cv2.boxFilter(l_toned * l_toned, cv2.CV_32F, (9, 9))
        local_std = np.sqrt(np.maximum(0.0, mean_ll - mean_l * mean_l))
        struct_gate = np.clip((local_std - 5.0) / 8.0, 0.0, 1.0)

        clahe = cv2.createCLAHE(clipLimit=1.30, tileGridSize=(8, 8))
        l_clahe = clahe.apply(np.clip(l_toned, 0, 255).astype(np.uint8)).astype(np.float32)
        l_hd = l_toned * (1.0 - 0.25 * struct_gate) + l_clahe * (0.25 * struct_gate)

        blur_fine = cv2.GaussianBlur(l_hd, (0, 0), 1.0)
        d_fine = np.clip(l_hd - blur_fine, -14.0, 14.0)
        l_hd = np.clip(l_hd + d_fine * 0.32 * struct_gate, 0.0, 255.0)

        return l_hd, h, w

    @classmethod
    def harmonize_ddcolor_with_orig_luminance(
        cls,
        orig_bgr: np.ndarray,
        ddc_bgr: np.ndarray,
        sr_bgr: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        l_base, h, w = cls._reconstruct_hd_luminance(orig_bgr, sr_bgr=sr_bgr)

        if ddc_bgr.shape[:2] != (h, w):
            ddc_bgr = cv2.resize(ddc_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

        ddc_lab = cv2.cvtColor(ddc_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_raw_u8 = cv2.medianBlur(np.clip(ddc_lab[:, :, 1], 0, 255).astype(np.uint8), 5).astype(np.float32)
        b_raw_u8 = cv2.medianBlur(np.clip(ddc_lab[:, :, 2], 0, 255).astype(np.uint8), 5).astype(np.float32)

        a_ch = cls._guided_filter_gray(l_base, a_raw_u8, radius=4, eps=18.0)
        b_ch = cls._guided_filter_gray(l_base, b_raw_u8, radius=4, eps=18.0)

        db_raw = b_ch - 128.0
        da_raw = a_ch - 128.0

        raw_overexp_sky = cls._detect_overexposed_sky_mask(np.clip(l_base, 0, 255).astype(np.uint8))
        ddc_not_warm_wall = np.clip((0.8 - db_raw) / 2.2, 0.0, 1.0)
        overexp_sky_mask = raw_overexp_sky * ddc_not_warm_wall

        yy = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
        upper_sky_zone = np.clip((yy - 0.52) / 0.20, 0.0, 1.0)
        sky_blue_mask = np.clip((-db_raw - 1.5) / 6.5, 0.0, 1.0) * upper_sky_zone * np.clip((l_base - 168.0) / 24.0, 0.0, 1.0)

        zenith_grad = np.linspace(1.0, 0.25, h, dtype=np.float32)[:, None]
        target_sky_l = 228.0 - 20.0 * zenith_grad
        l_final = l_base * (1.0 - 0.55 * overexp_sky_mask) + np.minimum(l_base, target_sky_l) * (0.55 * overexp_sky_mask)

        total_sky_mask = np.maximum(sky_blue_mask, overexp_sky_mask)
        non_sky_mask = 1.0 - np.clip(total_sky_mask * 1.35, 0.0, 1.0)

        non_sky_blue = non_sky_mask * np.clip(-db_raw / 2.2, 0.0, 1.0)
        db_fixed = np.where(
            db_raw < 0.0,
            db_raw * (0.90 * total_sky_mask + 0.28 * non_sky_mask),
            db_raw,
        )
        dark_roof_warm = non_sky_blue * np.clip((142.0 - l_final) / 38.0, 0.0, 1.0)
        db_work = db_fixed * (1.0 - dark_roof_warm) + 2.2 * dark_roof_warm
        da_work = da_raw * (1.0 - 0.5 * non_sky_blue) + 0.5 * dark_roof_warm

        non_sky_bool = non_sky_mask > 0.5
        if np.count_nonzero(non_sky_bool) > 100:
            pct_warm = float(np.mean(db_work[non_sky_bool] > 1.8))
            std_b = float(np.std(db_work[non_sky_bool]))
            if pct_warm > 0.88 and std_b < 3.4:
                sepia_b = float(np.percentile(db_work[non_sky_bool], 20))
                sepia_a = float(np.percentile(da_work[non_sky_bool], 20))
                if sepia_b > 2.2:
                    road_zone = np.clip((0.42 - yy) / 0.16, 0.0, 1.0)
                    road_l = np.clip(1.0 - np.abs(l_final - 122.0) / 42.0, 0.0, 1.0)
                    road_w = non_sky_mask * road_zone * road_l
                    db_work = db_work - sepia_b * road_w * 0.60
                    da_work = da_work - sepia_a * road_w * 0.60

        warm_expand = np.where(db_work > 0.0, db_work * 1.28, np.maximum(db_work, -11.0 * total_sky_mask - 3.2 * non_sky_mask))
        warm_cap = 0.85 * np.maximum(0.0, warm_expand) + 6.0
        max_pos_da = np.where(warm_expand > 0.0, warm_cap, 0.0)

        foliage_l_gate = np.clip((142.0 - l_final) / 26.0, 0.0, 1.0)
        bright_wall_gate = non_sky_mask * np.clip((l_final - 142.0) / 26.0, 0.0, 1.0) * np.clip(db_work / 3.0, 0.0, 1.0)
        foliage_mask = np.clip(-da_work / 2.0, 0.0, 1.0) * np.clip(db_work / 2.0, 0.0, 1.0) * foliage_l_gate
        da_neg_scaled = da_work * (1.28 * foliage_l_gate + 0.35 * (1.0 - foliage_l_gate)) + 1.4 * bright_wall_gate
        da_clean = np.where(da_work < 0.0, da_neg_scaled * (1.0 + 0.32 * foliage_mask), np.minimum(da_work * 1.24, max_pos_da))
        db_clean = warm_expand * (1.0 + 0.26 * foliage_mask)

        sky_target_da = -1.2 * zenith_grad
        sky_target_db = -8.5 * zenith_grad - 1.8
        da_clean = da_clean * (1.0 - 0.55 * overexp_sky_mask) + sky_target_da * (0.55 * overexp_sky_mask)
        db_clean = db_clean * (1.0 - 0.55 * overexp_sky_mask) + sky_target_db * (0.55 * overexp_sky_mask)

        da_harm = 21.0 * np.tanh(da_clean / 21.0)
        db_harm = 25.0 * np.tanh(db_clean / 25.0)

        gx = cv2.Sobel(l_final, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(l_final, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)
        edge_fringe_guard = 1.0 - 0.20 * np.clip((grad_mag - 50.0) / 80.0, 0.0, 1.0)

        shadow_atten = np.clip((l_final - 14.0) / 42.0, 0.34, 1.0)
        highlight_atten = np.clip((253.0 - l_final) / 16.0, 0.40, 1.0)
        chroma_scale = shadow_atten * highlight_atten * edge_fringe_guard

        a_final = 128.0 + da_harm * chroma_scale
        b_final = 128.0 + db_harm * chroma_scale

        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(np.clip(l_final, 0, 255).astype(np.uint8), scaleFactor=1.1, minNeighbors=4, minSize=(22, 22))
            if len(faces) > 0:
                skin_mask = np.zeros((h, w), dtype=np.float32)
                for (fx, fy, fw, fh) in faces:
                    cx, cy = fx + fw // 2, fy + fh // 2
                    axes = (max(6, int(fw * 0.44)), max(8, int(fh * 0.50)))
                    cv2.ellipse(skin_mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
                skin_mask = cv2.GaussianBlur(skin_mask, (0, 0), max(3.0, min(h, w) * 0.01))
                mid_skin = np.clip(1.0 - np.abs(l_final - 145.0) / 78.0, 0.0, 1.0)
                skin_w = skin_mask * mid_skin * 0.72
                a_final = a_final * (1.0 - skin_w) + (128.0 + 8.5) * skin_w
                b_final = b_final * (1.0 - skin_w) + (128.0 + 14.5) * skin_w
        except Exception:
            pass

        merged_lab = np.stack([
            np.clip(l_final, 0, 255).astype(np.uint8),
            np.clip(a_final, 0, 255).astype(np.uint8),
            np.clip(b_final, 0, 255).astype(np.uint8),
        ], axis=2)
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    def _get_ddc_client(self):
        from gradio_client import Client
        if self._ddc_client is None:
            with self._cloud_lock:
                if self._ddc_client is None:
                    self._ddc_client = Client("gudada/DDColor", token=self._get_hf_token(), verbose=False)
        return self._ddc_client

    def _get_opr_client(self):
        from gradio_client import Client
        if self._opr_client is None:
            with self._cloud_lock:
                if self._opr_client is None:
                    self._opr_client = Client("tarekmasryo/Old-Photo-Restorer", token=self._get_hf_token(), verbose=False)
        return self._opr_client

    @staticmethod
    def _restore_via_remote_colab_gpu(input_path: str, prompt_hint: str = "") -> Optional[np.ndarray]:
        if Path("/content").exists():
            return None
        colab_url = ""
        for p in (Path("colab_bridge_url.txt"),):
            if p.exists():
                colab_url = p.read_text(encoding="utf-8", errors="ignore").strip().rstrip("/")
                break
        if not colab_url:
            return None
        try:
            import base64

            b64_in = base64.b64encode(Path(input_path).read_bytes()).decode("utf-8")
            payload = {
                "secret": "patriot-bridge-2026",
                "image_b64": b64_in,
                "prompt_hint": prompt_hint,
            }
            req = urllib.request.Request(
                f"{colab_url}/api/restore_gpu",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "PatriotClient/1.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                out_b64 = data.get("restored_b64")
                if out_b64:
                    raw = base64.b64decode(out_b64)
                    arr = np.frombuffer(raw, dtype=np.uint8)
                    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[AI Restoration] Remote Colab GPU delegation notice: {e}", flush=True)
        return None

    def restore_with_ddcolor_cloud(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        import concurrent.futures
        import tempfile

        cleaned_bgr = self._clean_sky_tears_and_defects(img_bgr)
        ih, iw = cleaned_bgr.shape[:2]
        gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
        ry1, ry2 = int(ih * 0.10), int(ih * 0.84)
        rx1, rx2 = int(iw * 0.08), int(iw * 0.92)
        p1, p99 = np.percentile(gray[ry1:ry2, rx1:rx2], (1.0, 99.0))
        norm_gray = np.clip((gray.astype(np.float32) - p1) * 240.0 / max(1.0, p99 - p1) + 8.0, 0, 255).astype(np.uint8)
        neutral_bgr = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2BGR)
        if max(ih, iw) > 768:
            sc_ddc = 768.0 / float(max(ih, iw))
            neutral_bgr = cv2.resize(neutral_bgr, (int(round(iw * sc_ddc)), int(round(ih * sc_ddc))), interpolation=cv2.INTER_AREA)

        ts = f"{int(time.time() * 1000)}_{threading.get_ident()}"
        ascii_tmp = Path(tempfile.gettempdir()) / f"patriot_ddc_{ts}.jpg"
        _imwrite_bgr(str(ascii_tmp), neutral_bgr, quality=94)

        def _task_ddcolor() -> Optional[np.ndarray]:
            try:
                from gradio_client import handle_file
            except ImportError:
                return None
            for attempt in range(2):
                try:
                    client = self._get_ddc_client()
                    res = client.predict(handle_file(str(ascii_tmp)), api_name="/colorize")
                    out_path = res[1] if isinstance(res, (list, tuple)) and len(res) > 1 else (res[0] if isinstance(res, (list, tuple)) else res)
                    if isinstance(out_path, dict):
                        out_path = out_path.get("path")
                    if out_path and os.path.exists(str(out_path)):
                        return _imread_bgr(str(out_path))
                except Exception as e:
                    print(f"[AI Restoration] Cloud DDColor attempt {attempt + 1} notice: {e}", flush=True)
                    with self._cloud_lock:
                        self._ddc_client = None
            return None

        def _run_srvgg_bg(src_bgr: np.ndarray) -> Optional[np.ndarray]:
            if self.srvgg_upscaler is None or not TORCH_AVAILABLE:
                return None
            try:
                sh, sw = src_bgr.shape[:2]
                max_in = 360
                if max(sh, sw) > max_in:
                    sc = max_in / float(max(sh, sw))
                    in_bgr = cv2.resize(src_bgr, (max(32, int(round(sw * sc))), max(32, int(round(sh * sc)))), interpolation=cv2.INTER_AREA)
                else:
                    in_bgr = src_bgr
                rgb = cv2.cvtColor(in_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                tensor_in = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)
                dev = "cuda" if torch.cuda.is_available() else (DML_DEVICE if DML_DEVICE is not None else "cpu")
                try:
                    self.srvgg_upscaler.to(dev)
                    with torch.no_grad():
                        out_t = self.srvgg_upscaler(tensor_in.to(dev)).clamp(0.0, 1.0).cpu().squeeze(0).numpy()
                finally:
                    if str(dev) != "cpu":
                        self.srvgg_upscaler.to("cpu")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                out_rgb = (np.transpose(out_t, (1, 2, 0)) * 255.0).round().astype(np.uint8)
                return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[AI Restoration] SRVGG 4x notice: {e}", flush=True)
                return None

        def _task_sr() -> Optional[np.ndarray]:
            face_bgr = None
            try:
                from gradio_client import handle_file
                client_opr = self._get_opr_client()
                res_o = client_opr.predict(
                    image_in=handle_file(str(ascii_tmp)),
                    preset="Aggressive",
                    strength=0.72,
                    upscale=2,
                    only_center_face=False,
                    blend=0.12,
                    detail_boost=0.45,
                    out_format="JPG",
                    jpg_quality=95,
                    api_name="/enhance_single",
                )
                p_o = res_o[1] if isinstance(res_o, (list, tuple)) and len(res_o) > 1 else (res_o[0] if isinstance(res_o, (list, tuple)) else res_o)
                if isinstance(p_o, dict):
                    p_o = p_o.get("path")
                if p_o and os.path.exists(str(p_o)):
                    face_bgr = _imread_bgr(str(p_o))
            except Exception as e:
                print(f"[AI Restoration] Cloud Face Restoration notice: {e}", flush=True)
                with self._cloud_lock:
                    self._opr_client = None

            bg_sr = _run_srvgg_bg(neutral_bgr)
            if face_bgr is not None and bg_sr is not None:
                fh, fw = face_bgr.shape[:2]
                bg_matched = cv2.resize(bg_sr, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
                base_up = cv2.resize(neutral_bgr, (fw, fh), interpolation=cv2.INTER_CUBIC)
                diff_f = np.max(np.abs(face_bgr.astype(np.float32) - base_up.astype(np.float32)), axis=2)
                f_mask = np.clip((diff_f - 4.0) / 14.0, 0.0, 1.0)
                f_mask = cv2.GaussianBlur(f_mask, (15, 15), 4.0)[:, :, None]
                return (bg_matched.astype(np.float32) * (1.0 - f_mask) + face_bgr.astype(np.float32) * f_mask).astype(np.uint8)
            return face_bgr if face_bgr is not None else bg_sr

        colored = None
        sr_bgr = None
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            fut_ddc = ex.submit(_task_ddcolor)
            fut_sr = ex.submit(_task_sr)
            try:
                colored = fut_ddc.result(timeout=26.0)
            except Exception as e:
                print(f"[AI Restoration] DDColor timeout/err: {repr(e)}", flush=True)
            try:
                sr_bgr = fut_sr.result(timeout=22.0)
            except Exception as e:
                print(f"[AI Restoration] SR timeout/err: {repr(e)}", flush=True)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
            ascii_tmp.unlink(missing_ok=True)

        if colored is not None:
            return self.harmonize_ddcolor_with_orig_luminance(cleaned_bgr, colored, sr_bgr=sr_bgr)
        if sr_bgr is not None:
            return self.colorize_seed(sr_bgr)
        return None

    def _enhance_faces_cpu(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        if self.gfpgan_cpu_net is None or not TORCH_AVAILABLE:
            return img_bgr, False
        try:
            h, w = img_bgr.shape[:2]
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            min_f = max(42, int(min(h, w) * 0.08))
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(min_f, min_f))
            if len(faces) == 0:
                return img_bgr, False

            faces_sorted = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[:2]
            out_bgr = img_bgr.copy()
            enhanced_any = False

            for (fx, fy, fw, fh) in faces_sorted:
                pad_x = int(fw * 0.45)
                pad_y = int(fh * 0.50)
                x1 = max(0, fx - pad_x)
                y1 = max(0, fy - pad_y)
                x2 = min(w, fx + fw + pad_x)
                y2 = min(h, fy + fh + pad_y)
                crop = out_bgr[y1:y2, x1:x2]
                ch, cw = crop.shape[:2]
                if ch < 48 or cw < 48:
                    continue

                crop_512 = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR)
                rgb_in = cv2.cvtColor(crop_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                rgb_norm = (rgb_in - 0.5) / 0.5
                tensor_in = torch.from_numpy(np.transpose(rgb_norm, (2, 0, 1))).unsqueeze(0)

                with torch.inference_mode():
                    out_t, _ = self.gfpgan_cpu_net(tensor_in, weight=0.5, randomize_noise=False)
                out_np = out_t.squeeze(0).cpu().numpy()
                out_np = np.clip((out_np * 0.5 + 0.5) * 255.0, 0, 255).round().astype(np.uint8)
                restored_512 = cv2.cvtColor(np.transpose(out_np, (1, 2, 0)), cv2.COLOR_RGB2BGR)
                restored_crop = cv2.resize(restored_512, (cw, ch), interpolation=cv2.INTER_LANCZOS4)

                mask = np.zeros((ch, cw), dtype=np.float32)
                cv2.ellipse(
                    mask,
                    (cw // 2, ch // 2),
                    (max(8, int(cw * 0.36)), max(8, int(ch * 0.40))),
                    0,
                    0,
                    360,
                    1.0,
                    -1,
                )
                mask = cv2.GaussianBlur(mask, (0, 0), max(4.0, min(ch, cw) * 0.08))[:, :, None]
                out_bgr[y1:y2, x1:x2] = (
                    crop.astype(np.float32) * (1.0 - mask * 0.72) + restored_crop.astype(np.float32) * (mask * 0.72)
                ).round().astype(np.uint8)
                enhanced_any = True

            return out_bgr, enhanced_any
        except Exception as e:
            print(f"[AI Restoration] CPU GFPGAN face enhance notice: {e}", flush=True)
            return img_bgr, False

    def restore_with_local_cpu_ai(self, img_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, str]]:
        if not TORCH_AVAILABLE or self.ddcolor_ms is None:
            return None

        with self._cpu_infer_lock:
            try:
                cleaned_bgr = self._clean_sky_tears_and_defects(img_bgr)
                ih, iw = cleaned_bgr.shape[:2]
                stages = ["Sky/Scratch Inpaint"]

                ddc_bgr = self.ddcolor_ms.process(cleaned_bgr)
                stages.append("DDColor Transformer (CPU)")

                target_long = 1152
                scale = target_long / float(max(ih, iw))
                tw = max(64, int(round(iw * scale)))
                th = max(64, int(round(ih * scale)))

                g_clean = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
                g_den = cv2.fastNlMeansDenoising(g_clean, None, h=2.8, templateWindowSize=7, searchWindowSize=21)
                g_hd = cv2.resize(g_den, (tw, th), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)

                if self.srvgg_upscaler is not None and max(ih, iw) < 340:
                    try:
                        rgb_small = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        t_in = torch.from_numpy(np.transpose(rgb_small, (2, 0, 1))).unsqueeze(0)
                        with torch.inference_mode():
                            out_t = self.srvgg_upscaler(t_in).clamp(0.0, 1.0).squeeze(0).numpy()
                        sr_rgb = (np.transpose(out_t, (1, 2, 0)) * 255.0).round().astype(np.uint8)
                        sr_gray = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2GRAY)
                        sr_hd = cv2.resize(sr_gray, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
                        g_hd = g_hd * 0.70 + sr_hd * 0.30
                        stages.append("Real-ESRGAN SRVGG 4x (CPU)")
                    except Exception:
                        pass

                sky_mask = self._detect_overexposed_sky_mask(np.clip(g_hd, 0, 255).astype(np.uint8))
                if float(np.mean(sky_mask)) > 0.02:
                    g_sky_smooth = cv2.GaussianBlur(g_hd, (15, 15), 4.0)
                    g_hd = g_hd * (1.0 - 0.65 * sky_mask) + g_sky_smooth * (0.65 * sky_mask)

                inner_roi = g_hd[int(th * 0.08):int(th * 0.88), int(tw * 0.08):int(tw * 0.92)]
                p1, p99 = np.percentile(inner_roi, (1.0, 99.2))
                if p99 - p1 < 45.0:
                    p1, p99 = np.percentile(g_hd, (0.5, 99.5))
                l_01 = np.clip((g_hd - p1) / max(1.0, p99 - p1), 0.0, 1.0)
                l_toned = np.clip(np.power(l_01, 0.92) * 244.0 + 5.0, 0.0, 255.0).astype(np.float32)

                clahe = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8))
                l_clahe = clahe.apply(l_toned.astype(np.uint8)).astype(np.float32)
                l_hd = l_toned * (1.0 - 0.22 * (1.0 - sky_mask)) + l_clahe * (0.22 * (1.0 - sky_mask))

                blur_fine = cv2.GaussianBlur(l_hd, (0, 0), 1.0)
                l_hd = np.clip(l_hd + np.clip(l_hd - blur_fine, -12.0, 12.0) * 0.35 * (1.0 - sky_mask), 0.0, 255.0)

                ddc_up = cv2.resize(ddc_bgr, (tw, th), interpolation=cv2.INTER_CUBIC)
                ddc_lab = cv2.cvtColor(ddc_up, cv2.COLOR_BGR2LAB).astype(np.float32)
                a_guided = self._guided_filter_gray(l_hd, ddc_lab[:, :, 1], radius=3, eps=14.0)
                b_guided = self._guided_filter_gray(l_hd, ddc_lab[:, :, 2], radius=3, eps=14.0)

                da = 28.0 * np.tanh(((a_guided - 128.0) * 1.18) / 28.0)
                db = 32.0 * np.tanh(((b_guided - 128.0) * 1.18) / 32.0)

                if float(np.mean(sky_mask)) > 0.015:
                    zenith = np.linspace(1.0, 0.25, th, dtype=np.float32)[:, None]
                    not_warm = np.clip((1.5 - db) / 3.0, 0.0, 1.0)
                    eff_sky = sky_mask * not_warm
                    target_sky_l = 226.0 - 18.0 * zenith
                    l_hd = l_hd * (1.0 - 0.55 * eff_sky) + np.minimum(l_hd, target_sky_l) * (0.55 * eff_sky)
                    da = da * (1.0 - 0.60 * eff_sky) + (-1.2 * zenith) * (0.60 * eff_sky)
                    db = db * (1.0 - 0.65 * eff_sky) + (-9.5 * zenith - 2.0) * (0.65 * eff_sky)

                out_lab = np.stack([
                    np.clip(l_hd, 0, 255).astype(np.uint8),
                    np.clip(128.0 + da, 0, 255).astype(np.uint8),
                    np.clip(128.0 + db, 0, 255).astype(np.uint8),
                ], axis=2)
                final_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
                stages.append("Guided HD Fusion (1152px)")
                return final_bgr, " + ".join(stages)
            except Exception as e:
                print(f"[AI Restoration] Local CPU DDColor error: {e}", flush=True)
                return None


    def restore_photo(self, input_path: str, output_path: str, prompt_hint: str = "") -> Dict[str, str]:
        self.lazy_init_base()
        t0 = time.time()

        img = _imread_bgr(input_path)
        if img is None:
            raise ValueError(f"Не удалось прочитать изображение: {input_path}")

        h, w = img.shape[:2]
        max_dim = 1280
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        if DEVICE == "cpu":
            local_res = self.restore_with_local_cpu_ai(img)
            if local_res is not None:
                final_local_bgr, pipeline_label = local_res
                try:
                    stem = Path(input_path).stem
                    _imwrite_bgr(str(Path("static/uploads") / f"dslr_bank_{stem}.webp"), final_local_bgr, quality=92)
                except Exception:
                    pass
                if output_path.lower().endswith(".webp"):
                    _save_webp_with_thumb(final_local_bgr, output_path, max_width=1152, quality=92, thumb_width=200, thumb_quality=72)
                else:
                    _imwrite_bgr(output_path, final_local_bgr, quality=94)
                elapsed = round(time.time() - t0, 2)
                return {
                    "elapsed_sec": str(elapsed),
                    "device": "Local CPU (8 Threads)",
                    "pipeline": pipeline_label,
                }

        scene_desc = self._describe_scene_with_gemini(input_path) or prompt_hint
        if TORCH_AVAILABLE and DEVICE == "cuda":
            cleaned = self._clean_sky_tears_and_defects(img)
            seeded = self.colorize_seed(cleaned)
            engines = ["Sky Tear Inpaint", "DDColor Transformer (CUDA)"]
            generated = None
            try:
                generated = self.generate_with_controlnet(seeded, scene_description=scene_desc)
                if generated is not None:
                    engines.append("Realistic Vision V6.0 + ControlNet Tile (CUDA FP16)")
                    if self.gfpgan_restorer is not None:
                        engines.append("GFPGAN v1.4 (2x Face-Mask Fusion)")
            except Exception as e:
                print(f"[AI Restoration] Colab CUDA Diffusion error: {e}", flush=True)

            final_colab = generated if generated is not None else seeded
            try:
                stem = Path(input_path).stem
                _imwrite_bgr(str(Path("static/uploads") / f"dslr_bank_{stem}.webp"), final_colab, quality=92)
            except Exception:
                pass
            if output_path.lower().endswith(".webp"):
                _save_webp_with_thumb(final_colab, output_path, max_width=1152, quality=92, thumb_width=200, thumb_quality=72)
            else:
                _imwrite_bgr(output_path, final_colab, quality=94)
            elapsed = round(time.time() - t0, 2)
            return {
                "elapsed_sec": str(elapsed),
                "device": "Colab Tesla T4 CUDA",
                "pipeline": " + ".join(engines),
            }

        cleaned = self._clean_sky_tears_and_defects(img)
        seeded = self.colorize_seed(cleaned)
        if output_path.lower().endswith(".webp"):
            _save_webp_with_thumb(seeded, output_path, max_width=1152, quality=92, thumb_width=200, thumb_quality=72)
        else:
            _imwrite_bgr(output_path, seeded, quality=94)

        elapsed = round(time.time() - t0, 2)
        return {
            "elapsed_sec": str(elapsed),
            "device": "Local CPU",
            "pipeline": "DDColor / LAB Restoration",
        }


restoration_pipeline = NeuralRestorationPipeline()


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
WIKI_USER_AGENT = "PatriotHistoricalApp/1.0 (https://t.me/patriot_by_putevod_bot)"


def _is_bitmap_url(u: str) -> bool:
    if not u:
        return False
    base = u.split("?")[0].lower()
    return base.endswith((".jpg", ".jpeg", ".png", ".webp"))


BELARUS_CITY_ALIASES: Dict[str, List[str]] = {
    "гомель": ["гомел", "гомель", "homel", "gomel", "homyel", "homiel"],
    "светлогорск": ["светлогорск", "светлагорск", "svietlahorsk", "svetlahorsk", "шацілкі", "шатилки", "šacilki", "shatilki"],
    "минск": ["минск", "мінск", "minsk", "miensk"],
    "брест": ["брест", "брэст", "brest", "berastse"],
    "гродно": ["гродн", "гародн", "hrodna", "grodno"],
    "витебск": ["витебск", "віцебск", "vitebsk", "viciebsk"],
    "могилёв": ["могилев", "могилёв", "магілёў", "mahiliou", "mogilev"],
    "могилев": ["могилев", "могилёв", "магілёў", "mahiliou", "mogilev"],
    "полоцк": ["полоцк", "полацк", "polatsk", "polotsk"],
    "мозырь": ["мозыр", "мазыр", "mazyr", "mozyr"],
    "речица": ["речиц", "рэчыц", "rechytsa", "rechitsa"],
    "жлобин": ["жлобин", "жлобін", "zhlobin", "žlobin"],
    "бобруйск": ["бобруйск", "бабруйск", "babruysk", "bobruisk"],
    "барановичи": ["баранович", "баранавіч", "baranovichi", "baranavichy", "baranavičy"],
    "клецк": ["клецк", "kletsk", "kleck"],
    "несвиж": ["несвиж", "нясвіж", "niasvizh", "nesvizh"],
    "мир": ["мирск", "мірск", "mir castle", "мір"],
    "туров": ["туров", "тураў", "turau", "turov"],
    "пинск": ["пинск", "пінск", "pinsk"],
    "лида": ["лид", "лід", "lida"],
    "новогрудок": ["новогруд", "навагруд", "navahrudak", "novogrudok"],
    "орша": ["орш", "орша", "orsha"],
    "борисов": ["борисов", "барысаў", "barysaw", "borisov"],
    "слуцк": ["слуцк", "slutsk"],
    "слоним": ["слоним", "слонім", "slonim"],
    "кобрин": ["кобрин", "кобрын", "kobrin", "kobryn"],
    "волковыск": ["волковыск", "ваўкавыск", "volkovysk", "vawkavysk"],
    "молодечно": ["молодечн", "маладзечн", "molodechno", "maladzyechna"],
    "солигорск": ["солигорск", "салігорск", "soligorsk", "salihorsk"],
    "заславль": ["заслав", "заслаў", "zaslavl", "zaslawye"],
    "глубокое": ["глубок", "глыбок", "glubokoe", "hlybokaye"],
    "браслав": ["браслав", "браслаў", "braslav", "braslaw"],
    "поставы": ["постав", "пастав", "postavy", "pastavy"],
    "мстиславль": ["мстислав", "мсціслаў", "mstislavl"],
}

FOREIGN_OR_INVALID_TOKENS = [
    "калининград", "kaliningrad", "rauschen", "раушен",
    "крым", "crimea", "керчь", "kerch", "севастопол", "sevastopol",
    "брянск", "bryansk", "тула", "tula", "калуг", "воронеж", "самар", "екатеринбург",
    "украин", "ukraine", "польш", "poland",
    "coat of arms", "герб", "flag of", "флаг ", "stamp", "почтовая марка",
    "map of", "карта ", "схема ", "diagram", "logo", "icon", "locator",
    "pigcombinat", "свинокомплекс", "хрущёвок", "quercus", "dacha",
]

_STOP_STEM_WORDS = {
    "памятник", "мемориал", "мемориальный", "комплекс", "братская", "могила",
    "музей", "исторический", "краеведческий", "областной", "районный", "городской",
    "город", "города", "улица", "площадь", "здание",
    "дом", "бывший", "бывшая", "бывшее", "имени", "честь", "знак", "памятный",
    "святого", "святых", "свято", "святой", "памяти", "воинам", "советским",
    "беларусь", "белорусский", "республика", "гомельский", "светлогорский",
    "бобруйский", "клецкий", "барановичский", "полоцкий", "мозырский",
    "культуры", "отдыха", "летия", "ссср", "великой", "отечественной", "войны",
    "пресвятой", "богородицы", "божией", "божьей", "матери", "иконы",
    "благоверного", "князя", "великомученика", "храм", "церковь", "собор", "костел", "костёл",
}

_ARCHIVAL_GENERIC_STEMS = {
    "парк", "сквер", "аллея", "монаст", "часовн", "музе", "истор", "школ",
    "народн", "защитн", "узник", "гетто", "вечн", "огон", "славы", "побед",
    "депо", "пожарн", "борис", "симеон", "пушкар", "природ", "эколог", "средневек",
}


def _extract_distinctive_stems(title: str, city: str = "") -> List[str]:
    clean = re.sub(r"[^а-яёa-z0-9\s-]", " ", (title or "").lower().replace("ё", "е"))
    city_low = (city or "").lower().replace("ё", "е").strip()
    city_stem = city_low[:4] if len(city_low) >= 4 else city_low
    stems: List[str] = []
    fallback_type_stems: List[str] = []
    for raw_w in re.split(r"[\s-]+", clean):
        if len(raw_w) < 4:
            continue
        if city_stem and raw_w.startswith(city_stem):
            continue
        if raw_w.isdigit():
            continue
        stem = raw_w[:5] if len(raw_w) >= 6 else raw_w[:4]
        if raw_w in ("парк", "сквер", "аллея", "храм", "церковь", "собор", "костел", "музей"):
            if stem not in fallback_type_stems:
                fallback_type_stems.append(stem)
            continue
        if raw_w in _STOP_STEM_WORDS:
            continue
        if stem not in stems:
            stems.append(stem)
    return stems if stems else fallback_type_stems


def _does_archival_match_landmark(
    arch_caption: str,
    landmark_title: str,
    city: str = "",
    landmark_desc: str = "",
    dist_km: Optional[float] = None,
) -> bool:
    if not arch_caption or not landmark_title:
        return False

    arch_body = re.sub(r"^архивный\s+(?:снимок|кадр|исторический\s+снимок)\s*(?:\([^)]*\))?\s*:?\s*", "", arch_caption.strip(), flags=re.IGNORECASE)
    arch_low = arch_body.lower().replace("ё", "е")
    title_low = landmark_title.lower().replace("ё", "е")
    city_low = (city or "").lower().replace("ё", "е").strip()

    if city_low:
        for alias in BELARUS_CITY_ALIASES.get(city_low, [city_low[:5]]):
            arch_low = arch_low.replace(alias, " ")

    if "костел" in title_low and ("собор" in arch_low or "церков" in arch_low) and "костел" not in arch_low and "фарн" not in arch_low:
        return False
    if ("собор" in title_low or "церков" in title_low or "храм" in title_low) and "костел" in arch_low and "бывш" not in title_low and "воскресен" not in title_low:
        return False
    if any(w in arch_low for w in ("школа", "дом пионеров", "дом физкультуры", "горисполком", "райисполком", "универмаг", "аптека", "кинотеатр", "общежитие", "пту ")):
        if not any(w in title_low for w in ("школа", "пионеров", "физкультуры", "исполком", "универмаг", "аптека", "кинотеатр")):
            return False

    if "евфрос" in title_low or "ефрос" in title_low:
        return any(s in arch_low for s in ("евфрос", "ефрос", "спасо-е", "св.спаса", "святого спаса"))
    if "воскресен" in title_low and "клецк" in city_low:
        return any(s in arch_low for s in ("воскресен", "благовещ", "доминикан"))
    if "троиц" in title_low and "клецк" in city_low:
        return any(s in arch_low for s in ("троиц", "фарн"))
    if "миренбург" in title_low or "муравьев" in title_low:
        return any(s in arch_low for s in ("миренбург", "муравьев", "социалистич"))
    if "бахаров" in title_low or "танк" in title_low:
        return any(s in arch_low for s in ("танк", "бахаров"))
    if "железнодорож" in title_low:
        return any(s in arch_low for s in ("железнодорож", "станция", "вокзал", "паровоз"))
    if "часовня" in title_low or "усыпальниц" in title_low:
        return any(s in arch_low for s in ("часовн", "усыпальн"))
    if "петропавлов" in title_low:
        return any(s in arch_low for s in ("петропавл", "петра и павла"))
    if "борисов камень" in title_low:
        return "борис" in arch_low and "камен" in arch_low

    stems = [
        st for st in _extract_distinctive_stems(landmark_title, city=city)
        if st not in _ARCHIVAL_GENERIC_STEMS and len(st) >= 4
    ]
    if not stems:
        return False

    if any(st in arch_low for st in stems):
        return True

    try:
        from vector_engine import compute_archival_landmark_similarity

        vec_sim = compute_archival_landmark_similarity(
            arch_caption=arch_caption,
            landmark_title=landmark_title,
            landmark_desc=landmark_desc,
            city=city,
            dist_km=dist_km,
        )
        if vec_sim >= 0.56 and any(st[:4] in arch_low for st in stems):
            return True
    except Exception:
        pass

    return False


def _extract_canonical_queries(title: str) -> List[str]:
    no_parens = re.sub(r"\([^)]*\)", "", title or "").strip()
    no_quotes = no_parens.replace("«", "").replace("»", "").replace('"', "").strip()
    queries: List[str] = []
    if no_quotes:
        queries.append(no_quotes)

    for q_match in re.findall(r"«([^»]{3,})»", title or ""):
        q_clean = q_match.strip()
        if q_clean and q_clean not in queries:
            queries.append(q_clean)

    for sep in (" — ", " / ", " и "):
        if sep in no_quotes:
            parts = [p.strip() for p in no_quotes.split(sep) if len(p.strip()) >= 5]
            for p in reversed(parts):
                if p not in queries:
                    queries.append(p)

    if "святых " in no_quotes.lower():
        alt = re.sub(r"\b[Сс]вятых\s+", "", no_quotes).strip()
        if alt and alt not in queries:
            queries.append(alt)

    return queries[:3]


def _verify_candidate_metadata(meta_text: str, city: str, location_title: str = "", allow_untagged_geo: bool = False) -> bool:
    low = (meta_text or "").lower()
    if not low:
        return allow_untagged_geo

    for bad in FOREIGN_OR_INVALID_TOKENS:
        if bad in low:
            return False

    if "площадь труда" in low:
        m = re.search(r"площадь труда\s+(\d+)", low)
        if m and m.group(1) not in ("01", "1", "12"):
            return False

    city_clean = (city or "").strip().lower()
    city_ok = True
    if city_clean:
        aliases = BELARUS_CITY_ALIASES.get(city_clean, [city_clean[:5] if len(city_clean) >= 5 else city_clean])
        if any(a in low for a in aliases):
            city_ok = True
        elif allow_untagged_geo and ("беларус" in low or "belarus" in low or "by" in low):
            city_ok = True
        else:
            city_ok = allow_untagged_geo

    if not city_ok:
        return False

    if location_title:
        stems = _extract_distinctive_stems(location_title, city=city)
        if stems and not any(st in low for st in stems):
            return False

    return True


def _verify_photo_visual_quality(img_bgr: np.ndarray, is_archival: bool = False) -> bool:
    if img_bgr is None or img_bgr.ndim != 3:
        return False
    h, w = img_bgr.shape[:2]
    if h < 220 or w < 220:
        return False
    aspect = w / float(h)
    if aspect < 0.45 or aspect > 2.55:
        return False

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < 22.0:
        return False

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    white_ratio = float(np.mean((val > 244.0) & (sat < 10.0)))
    max_white = 0.58 if is_archival else 0.32
    if white_ratio > max_white:
        return False

    if not is_archival:
        mean_sat = float(np.mean(sat))
        p95_sat = float(np.percentile(sat, 95))
        if mean_sat < 12.0 and p95_sat < 30.0:
            return False

    return True


def _fetch_commons_files_by_titles(titles: List[str]) -> List[str]:
    if not titles:
        return []
    norm_titles: List[str] = []
    for t in titles:
        clean_t = t.strip()
        if not clean_t.lower().startswith("file:"):
            clean_t = f"File:{clean_t}"
        norm_titles.append(clean_t)

    joined = "|".join(norm_titles[:20])
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query"
        f"&titles={urllib.parse.quote(joined)}"
        "&prop=imageinfo&iiprop=url&iiurlwidth=1024&format=json"
    )
    resolved_map: Dict[str, str] = {}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": WIKI_USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                ptitle = (page.get("title") or "").replace("_", " ").lower()
                ii = page.get("imageinfo", [])
                if ii:
                    u = ii[0].get("thumburl") or ii[0].get("url")
                    if u and _is_bitmap_url(u):
                        resolved_map[ptitle] = u
    except Exception as e:
        print(f"[Wikimedia Batch] Notice: {e}", flush=True)

    ordered_urls: List[str] = []
    for req_t in norm_titles:
        key = req_t.replace("_", " ").lower()
        u = resolved_map.get(key)
        if not u:
            stem = key.replace("file:", "").strip()
            for k, val in resolved_map.items():
                if stem in k or k.replace("file:", "").strip() in stem:
                    u = val
                    break
        if u and u not in ordered_urls:
            ordered_urls.append(u)
    return ordered_urls


def _search_wikipedia_images(query: str, city: str = "", lang: str = "ru", location_title: str = "") -> List[str]:
    q = urllib.parse.quote(query.strip())
    url = (
        f"https://{lang}.wikipedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={q}&gsrlimit=6&prop=pageimages|extracts&exintro=1&explaintext=1"
        f"&piprop=thumbnail|original&pithumbsize=1024&format=json"
    )
    results: List[str] = []
    try:
        time.sleep(0.15)
        req = urllib.request.Request(url, headers={"User-Agent": WIKI_USER_AGENT})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page in sorted(pages.values(), key=lambda x: x.get("index", 99)):
                meta = f"{page.get('title', '')} {page.get('extract', '')}"
                if not _verify_candidate_metadata(meta, city=city, location_title=location_title):
                    continue
                thumb = page.get("thumbnail", {}).get("source") or page.get("original", {}).get("source")
                if thumb and _is_bitmap_url(thumb) and not _verify_url_blacklist(thumb) and thumb not in results:
                    results.append(thumb)
    except Exception:
        pass
    return results


def _verify_url_blacklist(u: str) -> bool:
    low = urllib.parse.unquote(u).lower()
    return any(tok in low for tok in FOREIGN_OR_INVALID_TOKENS)


def _search_wikimedia_geosearch(
    lat: float,
    lon: float,
    city: str = "",
    radius: int = 350,
    limit: int = 8,
    location_title: str = "",
) -> List[str]:
    urls: List[str] = []
    for lang in ("be", "ru"):
        w_url = (
            f"https://{lang}.wikipedia.org/w/api.php?action=query&generator=geosearch"
            f"&ggscoord={lat}|{lon}&ggsradius={min(radius, 10000)}&ggslimit={limit}"
            "&prop=pageimages&piprop=thumbnail|original&pithumbsize=1024&format=json"
        )
        try:
            time.sleep(0.12)
            req = urllib.request.Request(w_url, headers={"User-Agent": WIKI_USER_AGENT})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    ptitle = page.get("title", "")
                    if not _verify_candidate_metadata(
                        ptitle,
                        city=city,
                        location_title=location_title if radius > 220 else "",
                        allow_untagged_geo=True,
                    ):
                        continue
                    u = page.get("thumbnail", {}).get("source") or page.get("original", {}).get("source")
                    if u and _is_bitmap_url(u) and not _verify_url_blacklist(u) and u not in urls:
                        urls.append(u)
        except Exception:
            pass

    c_url = (
        "https://commons.wikimedia.org/w/api.php?action=query&generator=geosearch"
        f"&ggscoord={lat}|{lon}&ggsradius={min(radius, 10000)}&ggsnamespace=6&ggslimit={limit}"
        "&prop=imageinfo&iiprop=url|extmetadata&iiurlwidth=1024&format=json"
    )
    try:
        time.sleep(0.15)
        req = urllib.request.Request(c_url, headers={"User-Agent": WIKI_USER_AGENT})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                ii = page.get("imageinfo", [])
                if not ii:
                    continue
                ext = ii[0].get("extmetadata", {})
                desc = (ext.get("ImageDescription", {}) or {}).get("value", "")
                cats = (ext.get("Categories", {}) or {}).get("value", "")
                meta = f"{page.get('title', '')} {desc} {cats}"
                if not _verify_candidate_metadata(
                    meta,
                    city=city,
                    location_title=location_title if radius > 220 else "",
                    allow_untagged_geo=True,
                ):
                    continue
                u = ii[0].get("thumburl") or ii[0].get("url")
                if u and _is_bitmap_url(u) and not _verify_url_blacklist(u) and u not in urls:
                    urls.append(u)
    except Exception:
        pass
    return urls


def _search_wikimedia_commons(
    query: str,
    city: str = "",
    limit: int = 8,
    allow_untagged: bool = False,
    location_title: str = "",
) -> List[str]:
    q = urllib.parse.quote(f"{query.strip()} filetype:bitmap")
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={q}&gsrnamespace=6&gsrlimit={limit}&prop=imageinfo&iiprop=url|extmetadata&iiurlwidth=1024&format=json"
    )
    urls: List[str] = []
    try:
        time.sleep(0.2)
        req = urllib.request.Request(url, headers={"User-Agent": WIKI_USER_AGENT})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page in sorted(pages.values(), key=lambda x: x.get("index", 99)):
                ii = page.get("imageinfo", [])
                if ii:
                    ext = ii[0].get("extmetadata", {})
                    desc = (ext.get("ImageDescription", {}) or {}).get("value", "")
                    cats = (ext.get("Categories", {}) or {}).get("value", "")
                    meta = f"{page.get('title', '')} {desc} {cats}"
                    if not _verify_candidate_metadata(
                        meta,
                        city=city,
                        location_title=location_title,
                        allow_untagged_geo=allow_untagged,
                    ):
                        continue
                    u = ii[0].get("thumburl") or ii[0].get("url")
                    if u and _is_bitmap_url(u) and not _verify_url_blacklist(u) and u not in urls:
                        urls.append(u)
    except Exception:
        pass
    return urls


def _download_image_bgr(url: str) -> Optional[np.ndarray]:
    if url.startswith("local:"):
        local_p = Path(url.replace("local:", "", 1).strip())
        if local_p.exists():
            return _imread_bgr(str(local_p))
        return None

    if url.startswith("commons:"):
        titles = [url.replace("commons:", "", 1).strip()]
        resolved = _fetch_commons_files_by_titles(titles)
        if not resolved:
            return None
        url = resolved[0]

    clean_url = url.split("?")[0] if "wikimedia.org" in url else url
    if "avatars.mds.yandex.net" in clean_url and clean_url.endswith("/XXXL"):
        clean_url = clean_url[:-5] + "/XL"

    is_wiki = "wikimedia.org" in clean_url or "wikipedia.org" in clean_url
    max_attempts = 2 if is_wiki else 1
    for attempt in range(max_attempts):
        try:
            if is_wiki:
                time.sleep(0.12 + attempt * 0.35)
                ua = WIKI_USER_AGENT
            else:
                ua = USER_AGENT
            req = urllib.request.Request(
                clean_url,
                headers={
                    "User-Agent": ua,
                    "Accept-Language": "ru-RU,ru;q=0.9,be;q=0.8,en;q=0.7",
                },
            )
            with urllib.request.urlopen(req, timeout=5.5) as resp:
                raw = resp.read()
                arr = np.frombuffer(raw, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None and img.shape[0] >= 220 and img.shape[1] >= 220:
                    if "pastvu.com" in clean_url and img.shape[0] > 260:
                        crop_px = max(22, int(round(img.shape[0] * 0.018)))
                        img = img[:-crop_px, :]
                    return img
        except Exception:
            if attempt == 0 and max_attempts > 1:
                continue
    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


_YM_REJECT_CATEGORIES = (
    "ритуальн", "изготовление памятников", "кладбищ", "магазин", "супермаркет",
    "кафе", "ресторан", "бар, паб", "суши-бар", "спортбар", "пиццер", "кофейн", "столовая",
    "гостиниц", "отель", "аптека", "отделение банка", "банкомат", "салон красоты",
    "парикмахер", "автосервис", "автомойка", "шиномонтаж", "автозапчаст",
    "стоматолог", "больниц", "поликлиник", "торговый центр", "рынок",
    "строительн", "ремонт обуви", "ремонт одежд", "азс", "заправк", "нотариус",
)


def _search_yandex_maps_photos(
    title: str,
    city: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    max_results: int = 8,
) -> List[Tuple[str, str]]:
    stems = _extract_distinctive_stems(title, city=city)
    queries = _extract_canonical_queries(title)
    city_low = (city or "").strip().lower()
    results: List[Tuple[str, str]] = []
    seen_urls = set()

    for q_part in queries:
        if len(results) >= 4:
            break
        query = f"Беларусь {city} {q_part}".strip()
        q_encoded = urllib.parse.quote(query)
        url = f"https://yandex.by/maps/?text={q_encoded}"
        candidates_poi: List[Tuple[int, float, str, List[Dict[str, Any]]]] = []

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept-Language": "ru-RU,ru;q=0.9,be;q=0.8,en;q=0.7",
                },
            )
            with urllib.request.urlopen(req, timeout=11) as resp:
                html_text = resp.read().decode("utf-8", errors="ignore")
            m = re.search(r'<script[^>]*class="state-view"[^>]*>(.*?)</script>', html_text, flags=re.DOTALL)
            if not m:
                continue
            data = json.loads(m.group(1))
            stack = data.get("stack", [])

            def walk(obj: Any) -> None:
                if isinstance(obj, dict):
                    poi_title = obj.get("title")
                    coords = obj.get("coordinates") or obj.get("point")
                    if isinstance(poi_title, str) and isinstance(coords, list) and len(coords) == 2:
                        try:
                            plon, plat = float(coords[0]), float(coords[1])
                        except Exception:
                            plon, plat = 0.0, 0.0
                        pt_low = poi_title.strip().lower()
                        if pt_low in (city_low, f"город {city_low}", f"г. {city_low}"):
                            return
                        addr = str(obj.get("address") or obj.get("fullAddress") or obj.get("description") or "")
                        if not _verify_candidate_metadata(f"{poi_title} {addr}", city=city, allow_untagged_geo=True):
                            return
                        cats = [str(c.get("name", "")).lower() for c in (obj.get("categories") or []) if isinstance(c, dict)]
                        cat_str = " | ".join(cats)
                        if any(rc in cat_str for rc in _YM_REJECT_CATEGORIES) or "ритуальн" in pt_low or "изготовление памятников" in pt_low:
                            return

                        combined_low = f"{pt_low} {cat_str} {addr.lower()}"
                        stem_hits = sum(1 for st in stems if st in combined_low) if stems else 1

                        dist = 0.0
                        if lat is not None and lon is not None and plat and plon:
                            dist = _haversine_km(lat, lon, plat, plon)
                            max_allowed_dist = 5.0
                            if dist > max_allowed_dist:
                                return

                        if stems and stem_hits == 0 and (lat is None or dist > 0.18):
                            return

                        photos_obj = obj.get("photos") or {}
                        items = photos_obj.get("items") or []
                        if items:
                            candidates_poi.append((stem_hits, -dist, poi_title.strip(), items))
                    for v in obj.values():
                        walk(v)
                elif isinstance(obj, list):
                    for el in obj:
                        walk(el)

            walk(stack)
            candidates_poi.sort(key=lambda x: (x[0], x[1], len(x[3])), reverse=True)
            for _, _, poi_t, items in candidates_poi:
                for idx_p, it in enumerate(items):
                    ut = it.get("urlTemplate", "") if isinstance(it, dict) else ""
                    if ut:
                        u = ut.replace("%s", "XXXL")
                        if u not in seen_urls:
                            seen_urls.add(u)
                            cap = f"{poi_t} ({city})" if idx_p == 0 else f"{poi_t} — вид #{idx_p + 1}"
                            results.append((u, cap))
                            if len(results) >= max_results:
                                return results
        except Exception as e:
            print(f"[Yandex Maps Photo Search] Note for '{query}': {e}")

    return results


def _search_pastvu_archival_photos(
    lat: Optional[float],
    lon: Optional[float],
    distance_m: int = 2000,
    limit: int = 8,
) -> List[Tuple[str, str]]:
    if lat is None or lon is None:
        return []
    params_json = json.dumps(
        {
            "geo": [round(float(lat), 6), round(float(lon), 6)],
            "distance": distance_m,
            "year2": 1986,
            "limit": limit,
        }
    )
    api_url = f"https://pastvu.com/api2?method=photo.giveNearestPhotos&params={urllib.parse.quote(params_json)}"
    out: List[Tuple[str, str]] = []
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            photos = (data.get("result") or {}).get("photos") or []
            for p in photos:
                f = p.get("file")
                if not f:
                    continue
                year = p.get("year", "")
                ptitle = (p.get("title") or "Исторический снимок").strip()
                caption = f"Архивный снимок ({year} г.): {ptitle}" if year else f"Архивный снимок: {ptitle}"
                out.append((f"https://pastvu.com/_p/a/{f}", caption))
    except Exception as e:
        print(f"[PastVu Archival Search] Note at ({lat}, {lon}): {e}")
    return out


def _is_genuine_archival_photo(img_bgr: np.ndarray, from_archival_query: bool = False) -> bool:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    white_ratio = float(np.mean((val > 244.0) & (sat < 10.0)))
    if white_ratio > (0.58 if from_archival_query else 0.32):
        return False

    mid_bright = sat[val >= 40.0]
    if mid_bright.size < 100:
        mid_bright = sat

    mean_sat = float(np.mean(mid_bright))
    p95_sat = float(np.percentile(mid_bright, 95))

    if mean_sat < 22.0 and p95_sat <= 45.0:
        return True
    if from_archival_query and (mean_sat < 42.0 or p95_sat <= 85.0):
        return True
    return False


def _image_signature(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)


def _is_duplicate_image(sig: np.ndarray, existing_sigs: List[np.ndarray]) -> bool:
    for ex in existing_sigs:
        diff = float(np.mean(np.abs(sig - ex)))
        if diff < 12.0:
            return True
    return False


GOMEL_GALLERY_FILES: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
    "palace": {
        "modern": [
            ("https://avatars.mds.yandex.net/get-altay/14823903/2a00000197b76db8fc7469b2c643f017542f/XXXL", "Дворец Румянцевых и Паскевичей — главный фасад"),
            ("https://avatars.mds.yandex.net/get-altay/16914582/2a00000197b76dc67ab9cd87595a0ccae904/XXXL", "Колоннада и часовая башня дворца Паскевичей"),
            ("https://avatars.mds.yandex.net/get-altay/16609617/2a000001986b51c62b8688fc7807d8a91f2e/XXXL", "Памятник Н. П. Румянцеву перед дворцом"),
            ("https://avatars.mds.yandex.net/get-altay/15283873/2a00000196e47e57202f72612fd6344248c1/XXXL", "Четырёхъярусная башня князя Паскевича"),
        ],
        "archival": [
            ("commons:Ruined Rumyantsev-Paskevich Palace in Gomel, 1943.jpg", "Архивный снимок (1943 г.): Дворец Румянцевых и Паскевичей после освобождения"),
            ("https://pastvu.com/_p/a/j/d/s/jdsibrnp091f7si3cq.jpg", "Архивный снимок (1860 г.): Парадный Колонный зал дворца И. Ф. Паскевича"),
        ],
    },
    "memorial": {
        "modern": [
            ("https://avatars.mds.yandex.net/get-altay/4079915/2a00000178da740df3498c903da2836b882a/XXXL", "Мемориал «Вечный огонь» на площади Труда"),
            ("https://avatars.mds.yandex.net/get-altay/11393517/2a0000018d82bd0d4f488fc44452962c3065/XXXL", "Монумент воину-освободителю и братская могила на площади Труда"),
            ("https://avatars.mds.yandex.net/get-altay/6057477/2a00000181578982327dce107e4d3e5d8e7c/XXXL", "Мемориальный ансамбль на площади Труда в Гомеле"),
            ("https://avatars.mds.yandex.net/get-altay/4441655/2a000001815788cb37f8932f682bae8ef854/XXXL", "Сквер у Вечного огня на площади Труда"),
        ],
        "archival": [
            ("https://pastvu.com/_p/a/l/1/e/l1eo55xo196ugxitza.png", "Архивный снимок (1984 г.): Почётный караул у Вечного огня на площади Труда"),
            ("https://pastvu.com/_p/a/k/v/9/kv9xl2zstogwgvj091.jpg", "Архивный снимок (1977 г.): Площадь Труда в Гомеле"),
        ],
    },
    "kurgan": {
        "modern": [
            ("https://avatars.mds.yandex.net/get-altay/16873746/2a00000198004bd0fb6b7663335a5b78d7f1/XXXL", "Курган Славы и Вечный огонь в Гомеле"),
            ("https://avatars.mds.yandex.net/get-altay/16161139/2a000001981d61e0fd3799d26623ad222df1/XXXL", "Мемориальная плита у подножия Кургана Славы"),
            ("https://avatars.mds.yandex.net/get-altay/2786941/2a00000170d3c37a2fbf1199f8bd3af0d1b1/XXXL", "Вечный огонь у Кургана Славы в парке Фестивальный"),
            ("https://avatars.mds.yandex.net/get-altay/14398723/2a000001966be7648939d4c1a0e00e5fa4c2/XXXL", "Аллея парка Фестивальный у Кургана Славы"),
        ],
        "archival": [
            ("https://pastvu.com/_p/a/1/n/s/1ns8fko9pib8m7iccy.jpg", "Архивный снимок (1963 г.): Речицкое шоссе в районе закладки мемориала"),
            ("https://pastvu.com/_p/a/z/3/7/z37xtqe3innwr9mo93.jpg", "Архивный снимок (1970 г.): Окрестности парка Фестивальный в Гомеле"),
        ],
    },
    "lost": {
        "modern": [
            ("https://avatars.mds.yandex.net/get-altay/918124/2a00000161ca594fc95268174cbd0cc17ead/XXXL", "Исторический квартал улицы Советской (бывшей Румянцевской)"),
            ("https://avatars.mds.yandex.net/get-altay/965007/2a00000162b23a8a28deb8984856f32a5e99/XXXL", "Перекрёсток улиц Советской и Крестьянской — место бывшего Троицкого собора"),
            ("https://avatars.mds.yandex.net/get-altay/10834132/2a0000018e02fee132c9a3386733f3af897c/XXXL", "Старинная архитектура бывшей гостиницы «Савой» на Румянцевской"),
        ],
        "archival": [
            ("commons:Homel St Trinity Church 1900s.jpg", "Архивный снимок (1900-е гг.): Утраченный Троицкий собор в Гомеле"),
            ("https://pastvu.com/_p/a/i/y/8/iy8qcbkx9e0fd1kjol.jpg", "Архивный снимок (1941 г.): Перспектива улицы Советской (бывш. Румянцевской)"),
            ("https://pastvu.com/_p/a/a/s/8/as8qzjjmqwg58o23px.jpg", "Архивный снимок (1946 г.): Историческая застройка улицы Советской"),
        ],
    },
    "park": {
        "modern": [
            ("https://core-pht-proxy.maps.yandex.ru/v1/photos/download?photo_id=jOJIjVnkOQzcI32M5RlpSw&image_size=XXXL", "Лебяжий пруд с домиком для лебедей в овраге речки Гомеюк"),
            ("https://core-pht-proxy.maps.yandex.ru/v1/photos/download?photo_id=27YGqvXhc2gtBIWGGyQmAA&image_size=XXXL", "Арочный мостик через Лебяжий пруд и набережная Сожа"),
            ("https://core-pht-proxy.maps.yandex.ru/v1/photos/download?photo_id=TBgnr_CRnvArCfETHClSsw&image_size=XXXL", "Вечерняя панорама Лебяжьего пруда в Гомельском парке"),
            ("commons:Гомель. Парк. Набережная. Фото 52.jpg", "Набережная реки Сож в старинном парке Гомеля"),
        ],
        "archival": [
            ("https://pastvu.com/_p/a/y/4/h/y4hq7xwkbi5c2ldyr9.jpg", "Архивный снимок (1985 г.): Лебяжий пруд и домик лебедей в Гомельском парке"),
            ("local:static/demo/ref_bank/aerial_1941_2_orig.jpg", "Архивный снимок (1941 г.): Историческая панорама Гомельского парка и реки Сож"),
        ],
    },
    "hunter": {
        "modern": [
            ("https://avatars.mds.yandex.net/get-altay/1349040/2a000001642abeace8db66f3464501429c86/XXXL", "Охотничий домик (1820 г., ул. Пушкина, 32) — главный фасад с колоннами"),
            ("https://avatars.mds.yandex.net/get-altay/14793643/2a00000195097c3afc176e1c80bef4f4074d/XXXL", "Охотничий домик зимой — шестиколонный портик в стиле ампир"),
            ("https://avatars.mds.yandex.net/get-altay/14920824/2a00000195097c57a760bda3b290728d5c16/XXXL", "Усадебный ансамбль Охотничьего домика на улице Пушкина"),
            ("https://avatars.mds.yandex.net/get-altay/6545454/2a0000018d08b8da83d52dd79337d3abd3cb/XXXL", "Экспозиция Музея истории города Гомеля в Охотничьем домике"),
        ],
        "archival": [],
    },
}

SVETLOGORSK_GALLERY_FILES: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
    "kolokol": {
        "modern": [
            ("https://avatars.mds.yandex.net/get-altay/15223195/2a00000194cdce14c2253576d56951d37c48/XXXL", "Мемориальный комплекс «Молчащий колокол» на набережной Березины"),
            ("https://avatars.mds.yandex.net/get-altay/892711/2a0000018d2b2004feefdf9d666a4761479d/XXXL", "Расколотый бронзовый колокол — символ памяти павших воинов"),
            ("https://avatars.mds.yandex.net/get-altay/247136/2a0000015b163c1fabe683c9c249fc0caec4/XXXL", "Барельефы мемориала «Колокол» в Светлогорске"),
            ("https://avatars.mds.yandex.net/get-altay/10232404/2a00000191925485c3934c885fe114580456/XXXL", "Мемориал «Колокол» на закате у реки Березины"),
        ],
        "archival": [],
    },
    "spaso": {
        "modern": [
            ("https://avatars.mds.yandex.net/get-altay/15408870/2a00000197efe306ddd56a3d0e3ca0c3e2f5/XXXL", "Храм Преображения Господня в Светлогорске — главный фасад"),
            ("https://avatars.mds.yandex.net/get-altay/223006/2a0000015b1636a63772382960aa1f5523a3/XXXL", "Пятиглавый Спасо-Преображенский собор в Светлогорске"),
            ("https://avatars.mds.yandex.net/get-altay/17684794/2a0000019e93520bd438200e5cae661bde9d/XXXL", "Спасо-Преображенский храм весной в цветущей сирени"),
            ("https://avatars.mds.yandex.net/get-altay/15586709/2a00000197efe31606ba0f5655aaf99d5b82/XXXL", "Внутреннее убранство и иконостас храма Преображения Господня"),
        ],
        "archival": [],
    },
    "kostel": {
        "modern": [
            ("https://avatars.mds.yandex.net/get-altay/14733519/2a00000197efbc576224767c520dd390efc5/XXXL", "Костёл Воздвижения Святого Креста в Светлогорске — неоготический фасад"),
            ("https://avatars.mds.yandex.net/get-altay/13671691/2a00000191925674f90a889998d2c0e4c273/XXXL", "Башни-колокольни костёла Воздвижения Святого Креста"),
            ("https://avatars.mds.yandex.net/get-altay/13897576/2a000001919256ebc9c515b60b44330e80ad/XXXL", "Архитектурный ансамбль костёла из красного кирпича"),
            ("https://avatars.mds.yandex.net/get-altay/18140291/2a0000019f08bf747dcafa657a2b91254973/XXXL", "Вид на костёл Воздвижения Святого Креста со стороны набережной"),
        ],
        "archival": [],
    },
    "shatilki": {
        "modern": [
            ("https://avatars.mds.yandex.net/get-altay/18105464/2a0000019e934fa024ddd2dc96be0335779a/XXXL", "Памятник основателю Шатилок Роману Шатиле в Светлогорске"),
            ("https://avatars.mds.yandex.net/get-altay/20308240/2a0000019f942af6edeebbd1171d67f2e497/XXXL", "Памятный знак 1560 года с привилеем Сигизмунда II Августа"),
            ("https://avatars.mds.yandex.net/get-altay/813485/2a0000015ece6efc641ae45dc5444f732570/XXXL", "Светлогорский историко-краеведческий музей и мемориальный валун"),
            ("commons:Stone with a cross from the Karaliova Slabada-2.JPG", "Королёвослободский камень с крестом XVII века у музея Светлогорска"),
        ],
        "archival": [],
    },
    "naberezhnaya": {
        "modern": [
            ("https://avatars.mds.yandex.net/get-altay/14922087/2a0000019541ff912c1d226ee20186985c79/XXXL", "Светлогорская картинная галерея «Традиция» им. Г. М. Прянишникова"),
            ("commons:Svetlahorsk Berezina river BY.jpg", "Панорама реки Березины и набережной в Светлогорске"),
            ("https://avatars.mds.yandex.net/get-altay/16411237/2a0000019c60234a7a300f934201994ece6c/XXXL", "Живописное полотно в экспозиции галереи «Традиция»"),
            ("https://avatars.mds.yandex.net/get-altay/19756885/2a0000019ebcfd1f0e9516d81c58a1685692/XXXL", "Скульптурная композиция у картинной галереи «Традиция»"),
        ],
        "archival": [],
    },
}


def build_standalone_archival_gallery(
    img_url: str,
    caption: str,
    slug: str,
    static_dir: str = "static",
) -> List[Dict[str, Any]]:
    photos_dir = Path(static_dir) / "real_photos"
    photos_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = photos_dir / f"{slug}_gallery.json"
    if manifest_file.exists():
        try:
            cached = json.loads(manifest_file.read_text(encoding="utf-8"))
            if isinstance(cached, list) and cached:
                f_rel = cached[0]["url"].lstrip("/").replace("static/", "", 1)
                if (Path(static_dir) / f_rel).exists():
                    return cached
        except Exception:
            pass

    img = _download_image_bgr(img_url)
    if img is None:
        return []
    fname = f"{slug}_arch1.webp"
    thumb_name = f"{slug}_arch1_thumb.webp"
    fpath = photos_dir / fname
    _save_webp_with_thumb(img, str(fpath), max_width=880, quality=82, thumb_width=200, thumb_quality=70)
    gallery = [
        {
            "url": f"/static/real_photos/{fname}",
            "thumb_url": f"/static/real_photos/{thumb_name}",
            "restored_url": None,
            "restored_thumb_url": None,
            "is_archival": True,
            "caption": caption,
        }
    ]
    manifest_file.write_text(json.dumps(gallery, ensure_ascii=False), encoding="utf-8")
    return gallery


def find_location_photo_gallery(
    title: str,
    city: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    static_dir: str = "static",
    custom_slug: str = "",
    max_photos: int = 6,
    force_refresh: bool = False,
    preverified_photos: Optional[List[Tuple[str, str]]] = None,
    preverified_archival: Optional[List[Tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor

    photos_dir = Path(static_dir) / "real_photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    slug = custom_slug or hashlib.md5(f"{city}_{title}".encode("utf-8")).hexdigest()[:10]
    manifest_file = photos_dir / f"{slug}_gallery.json"

    if not force_refresh and manifest_file.exists():
        try:
            saved = json.loads(manifest_file.read_text(encoding="utf-8"))
            if isinstance(saved, list) and len(saved) >= 1:
                return saved
        except Exception:
            pass

    clean_title = re.sub(r"\([^)]*\)", "", title).strip()
    city_low = (city or "").strip().lower()
    tl = title.lower()

    modern_candidates: List[Tuple[str, str]] = []
    archival_candidates: List[Tuple[str, str]] = []

    if preverified_photos:
        for u, cap in preverified_photos:
            if u and not any(u == ex_u for ex_u, _ in modern_candidates):
                modern_candidates.append((u, cap))

    if preverified_archival is not None:
        for u, cap in preverified_archival:
            if u and not any(u == ex_u for ex_u, _ in archival_candidates):
                archival_candidates.append((u, cap))

    gomel_key = None
    if custom_slug.startswith("gomel_"):
        gomel_key = custom_slug.replace("gomel_", "")
    elif city_low == "гомель":
        if "румянцев" in tl and "дворец" in tl:
            gomel_key = "palace"
        elif "вечный огонь" in tl or "площади труда" in tl:
            gomel_key = "memorial"
        elif "курган славы" in tl:
            gomel_key = "kurgan"
        elif "троицкий собор" in tl or "исчезнувший" in tl:
            gomel_key = "lost"
        elif "лебяжий пруд" in tl:
            gomel_key = "park"
        elif "охотничий домик" in tl:
            gomel_key = "hunter"

    svet_key = None
    if city_low in ("светлогорск", "светлагорск", "svietlahorsk", "шатилки"):
        if "молчащий колокол" in tl or "комплекс «колокол»" in tl:
            svet_key = "kolokol"
        elif "преображения господня" in tl or "спасо-преображен" in tl:
            svet_key = "spaso"
        elif "воздвижения святого креста" in tl:
            svet_key = "kostel"
        elif "роману шатиле" in tl or "основателю шатилок" in tl or "королёвослободский" in tl:
            svet_key = "shatilki"
        elif "березины" in tl and "традиция" in tl:
            svet_key = "naberezhnaya"

    if gomel_key and gomel_key in GOMEL_GALLERY_FILES:
        cfg = GOMEL_GALLERY_FILES[gomel_key]
        modern_candidates.extend(cfg["modern"])
        archival_candidates.extend(cfg["archival"])
    elif svet_key and svet_key in SVETLOGORSK_GALLERY_FILES:
        cfg = SVETLOGORSK_GALLERY_FILES[svet_key]
        modern_candidates.extend(cfg["modern"])
        archival_candidates.extend(cfg["archival"])
    else:
        if len(modern_candidates) == 0:
            ym_photos = _search_yandex_maps_photos(title, city, lat=lat, lon=lon, max_results=6)
            for u, cap in ym_photos:
                if not any(u == ex_u for ex_u, _ in modern_candidates):
                    modern_candidates.append((u, cap))

        if lat is not None and lon is not None and len(modern_candidates) == 0:
            for u in _search_wikimedia_geosearch(lat, lon, city=city, radius=250, limit=5, location_title=clean_title):
                if not any(u == ex_u for ex_u, _ in modern_candidates):
                    modern_candidates.append((u, f"{clean_title} ({city})"))

        if len(modern_candidates) == 0:
            for cq in _extract_canonical_queries(title)[:1]:
                for u in _search_wikimedia_commons(f"{cq} {city}", city=city, limit=4, location_title=cq):
                    if not any(u == ex_u for ex_u, _ in modern_candidates):
                        modern_candidates.append((u, f"{clean_title} ({city})"))

        if preverified_archival is None and lat is not None and lon is not None:
            for pv_url, pv_cap in _search_pastvu_archival_photos(lat, lon, distance_m=2500, limit=12):
                if _does_archival_match_landmark(pv_cap, title, city=city):
                    if not any(pv_url == ex_u for ex_u, _ in archival_candidates):
                        archival_candidates.append((pv_url, pv_cap))

    gallery: List[Dict[str, Any]] = []
    signatures: List[np.ndarray] = []

    target_modern = max(1, max_photos - 1) if (archival_candidates and max_photos >= 3) else max(1, max_photos)
    mod_batch = modern_candidates[: target_modern + 2]
    if mod_batch:
        with ThreadPoolExecutor(max_workers=min(6, len(mod_batch))) as pool:
            downloaded_mod = list(pool.map(lambda item: (_download_image_bgr(item[0]), item[1]), mod_batch))
        for img, cap in downloaded_mod:
            if len(gallery) >= target_modern:
                break
            if img is None:
                continue
            if not _verify_photo_visual_quality(img, is_archival=False):
                continue
            sig = _image_signature(img)
            if _is_duplicate_image(sig, signatures):
                continue
            signatures.append(sig)

            idx = len(gallery) + 1
            fname = f"{slug}_p{idx}.webp"
            thumb_name = f"{slug}_p{idx}_thumb.webp"
            fpath = photos_dir / fname
            _save_webp_with_thumb(img, str(fpath), max_width=880, quality=80, thumb_width=200, thumb_quality=70)

            gallery.append(
                {
                    "url": f"/static/real_photos/{fname}",
                    "thumb_url": f"/static/real_photos/{thumb_name}",
                    "restored_url": None,
                    "restored_thumb_url": None,
                    "is_archival": False,
                    "caption": cap or f"{clean_title} — фото #{idx}",
                }
            )

    archival_added = 0
    arch_batch = archival_candidates[:4]
    if arch_batch:
        with ThreadPoolExecutor(max_workers=min(4, len(arch_batch))) as pool:
            downloaded_arch = list(pool.map(lambda item: (_download_image_bgr(item[0]), item[1]), arch_batch))
        for img, cap in downloaded_arch:
            if len(gallery) >= max_photos and archival_added >= 1:
                break
            if img is None:
                continue
            if not _is_genuine_archival_photo(img, from_archival_query=True):
                continue
            if not _verify_photo_visual_quality(img, is_archival=True):
                continue
            sig = _image_signature(img)
            if _is_duplicate_image(sig, signatures):
                continue
            signatures.append(sig)

            idx = len(gallery) + 1
            fname = f"{slug}_arch{idx}.webp"
            thumb_name = f"{slug}_arch{idx}_thumb.webp"
            fpath = photos_dir / fname
            _save_webp_with_thumb(img, str(fpath), max_width=880, quality=82, thumb_width=200, thumb_quality=70)

            gallery.append(
                {
                    "url": f"/static/real_photos/{fname}",
                    "thumb_url": f"/static/real_photos/{thumb_name}",
                    "restored_url": None,
                    "restored_thumb_url": None,
                    "is_archival": True,
                    "caption": cap or f"Архивный исторический снимок ({city})",
                }
            )
            archival_added += 1
            if archival_added >= 2:
                break

    if gallery:
        manifest_file.write_text(json.dumps(gallery, ensure_ascii=False), encoding="utf-8")
        print(
            f"[Photo Gallery] Verified & saved {len(gallery)} WebP photos ({archival_added} archival) for '{title} ({city})'",
            flush=True,
        )

    return gallery


def find_and_prepare_location_photos(
    title: str,
    city: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    static_dir: str = "static",
    custom_slug: str = "",
    variant_index: int = 0,
    run_gpu_ai: bool = False,
    force_refresh: bool = False,
) -> Tuple[str, str]:
    gallery = find_location_photo_gallery(
        title=title,
        city=city,
        lat=lat,
        lon=lon,
        static_dir=static_dir,
        custom_slug=custom_slug,
        force_refresh=force_refresh,
    )
    if not gallery:
        return "/static/demo/palace_before.webp", "/static/demo/palace_after.webp"

    modern_item = next((p for p in gallery if not p.get("is_archival")), gallery[0])
    archival_item = next((p for p in gallery if p.get("is_archival")), gallery[0])
    return archival_item["url"], modern_item["url"]


GOMEL_DEMO_TARGETS = {
    "palace": ("Дворец Румянцевых и Паскевичей", "Гомель"),
    "memorial": ("Мемориал «Вечный огонь» и братская могила на площади Труда", "Гомель"),
    "kurgan": ("Курган Славы и парк Фестивальный", "Гомель"),
    "lost": ("Исчезнувший Троицкий собор и старая Румянцевская (Советская) улица", "Гомель"),
    "park": ("Лебяжий пруд и набережная реки Сож в старинном парке", "Гомель"),
    "hunter": ("Охотничий домик (Летняя резиденция Городничего / Паскевичей)", "Гомель"),
}


def ensure_demo_images(static_dir: str = "static") -> None:
    demo_dir = Path(static_dir) / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    marker = demo_dir / ".real_gallery_webp_v7.marker"
    force = not marker.exists()

    for prefix, (title_q, city_q) in GOMEL_DEMO_TARGETS.items():
        gallery = find_location_photo_gallery(
            title=title_q,
            city=city_q,
            static_dir=static_dir,
            custom_slug=f"gomel_{prefix}",
            max_photos=6,
            force_refresh=force,
        )
        if gallery:
            modern_item = next((p for p in gallery if not p.get("is_archival")), gallery[0])
            archival_item = next((p for p in gallery if p.get("is_archival")), gallery[0])
            src_b = Path(static_dir) / archival_item["url"].lstrip("/").replace("static/", "", 1)
            src_a = Path(static_dir) / modern_item["url"].lstrip("/").replace("static/", "", 1)
            if src_b.exists() and src_a.exists():
                img_b = _imread_bgr(str(src_b))
                img_a = _imread_bgr(str(src_a))
                if img_b is not None and img_a is not None:
                    _save_webp_with_thumb(img_b, str(demo_dir / f"{prefix}_before.webp"))
                    _save_webp_with_thumb(img_a, str(demo_dir / f"{prefix}_after.webp"))

    marker.write_text("ok", encoding="utf-8")
