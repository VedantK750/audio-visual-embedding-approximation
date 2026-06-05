"""End-to-end inference footprint: params + FLOPs per clip for each full pipeline.

A "pipeline" is encoder(s) + student head (or, for the teacher, ImageBind alone).
We count what it actually costs to turn ONE clip into its 2048-d embedding:

    Teacher (1 frame)     : ImageBind(vision x1) + ImageBind(audio)
    Teacher (5 frame)     : ImageBind(vision x5) + ImageBind(audio)        (privileged gallery)
    MLP student           : CLIP(x1) + AST(x1) + NaiveLateFusionMLP
    Cross-attention       : SigLIP2(x1) + CLAP(x1) + CrossAttentionStudent
    Multi-token           : SigLIP2(x5) + CLAP(x5) + MultiTokenFusionTransformer

FLOPs come from torch's FlopCounterMode (multiply-add style; treat as relative).
Inputs are built from the real processors on one actual test clip, so shapes are
correct. Everything runs on CPU and each model is freed after measuring.

Run:  python3 tools/footprint.py
"""

import os
import sys
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image
import torchaudio
from torch.utils.flop_counter import FlopCounterMode

from avea.eval.extraction import gather_test_records
from avea.models.mlp_fusion import NaiveLateFusionMLP
from avea.models.transformer_fusion import CrossAttentionStudent
from avea.models.multitoken_fusion import MultiTokenFusionTransformer

DATA_ROOT = "processed_vggsound"
DEVICE = "cpu"


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def flops_of(fn):
    """Count FLOPs of a zero-arg callable that runs one forward."""
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        fn()
    return counter.get_total_flops()


def human(n):
    for unit in ["", "K", "M", "G", "T"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}P"


def measure_encoders(record):
    """Returns dict: name -> (params, flops_for_ONE_forward)."""
    out = {}
    mid = record["frame_paths"][len(record["frame_paths"]) // 2]
    audio_path = record["audio_path"]

    # ---- CLIP (image) + AST (audio): the MLP student's encoders ----
    from transformers import CLIPModel, CLIPProcessor, ASTModel, ASTFeatureExtractor
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    px = cproc(images=Image.open(mid).convert("RGB"), return_tensors="pt")["pixel_values"]
    out["CLIP image"] = (n_params(clip), flops_of(lambda: clip.get_image_features(pixel_values=px)))
    del clip, cproc; gc.collect()

    ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").eval()
    afe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    wav, sr = torchaudio.load(audio_path)
    feats = afe(wav.mean(0).numpy(), sampling_rate=16000, return_tensors="pt")
    out["AST audio"] = (n_params(ast), flops_of(lambda: ast(**feats)))
    del ast, afe; gc.collect()

    # ---- SigLIP 2 (image) + CLAP (audio): the transformer students' encoders ----
    from transformers import AutoModel, AutoProcessor, ClapModel, ClapProcessor
    siglip = AutoModel.from_pretrained("google/siglip2-base-patch16-224").eval()
    sproc = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    spx = sproc(images=Image.open(mid).convert("RGB"), return_tensors="pt")
    out["SigLIP2 image"] = (n_params(siglip), flops_of(lambda: siglip.get_image_features(**spx)))
    del siglip, sproc; gc.collect()

    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    clproc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    w48 = torchaudio.functional.resample(wav.mean(0), sr, 48000).numpy()
    cin = clproc(audio=w48, sampling_rate=48000, return_tensors="pt")
    out["CLAP audio"] = (n_params(clap), flops_of(lambda: clap.get_audio_features(**cin)))
    del clap, clproc; gc.collect()

    # ---- ImageBind teacher (vision trunk + audio trunk) ----
    try:
        from imagebind import data
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        ib = imagebind_model.imagebind_huge(pretrained=True).eval()
        vis1 = data.load_and_transform_vision_data([mid], DEVICE)
        vis5 = data.load_and_transform_vision_data(record["frame_paths"], DEVICE)
        aud = data.load_and_transform_audio_data([audio_path], DEVICE)
        ib_params = n_params(ib)
        out["ImageBind vision x1"] = (ib_params, flops_of(lambda: ib({ModalityType.VISION: vis1})))
        out["ImageBind vision x5"] = (ib_params, flops_of(lambda: ib({ModalityType.VISION: vis5})))
        out["ImageBind audio"] = (ib_params, flops_of(lambda: ib({ModalityType.AUDIO: aud})))
        del ib; gc.collect()
    except Exception as e:
        print(f"[warn] ImageBind measurement skipped: {e}")

    return out


def measure_heads():
    out = {}
    m = NaiveLateFusionMLP().eval()
    out["MLP head"] = (n_params(m), flops_of(lambda: m(torch.randn(1, 512), torch.randn(1, 768))))
    c = CrossAttentionStudent().eval()
    out["Cross-attn head"] = (n_params(c), flops_of(lambda: c(torch.randn(1, 768), torch.randn(1, 512))))
    t = MultiTokenFusionTransformer().eval()
    out["Multi-token head"] = (n_params(t), flops_of(lambda: t(torch.randn(1, 5, 768), torch.randn(1, 5, 512))))
    return out


def main():
    record = gather_test_records(DATA_ROOT)[0]
    print("Measuring (CPU, one clip)... this loads several large models, give it a minute.\n")

    enc = measure_encoders(record)
    heads = measure_heads()

    def comp_p(name):
        return enc.get(name, (0, 0))[0]

    def comp_f(name):
        return enc.get(name, (0, 0))[1]

    # (pipeline, unique-weight components for params, (component, multiplier) for flops, head)
    pipelines = [
        ("Teacher (1 frame)", ["ImageBind vision x1", "ImageBind audio"],
         [("ImageBind vision x1", 1), ("ImageBind audio", 1)], None),
        ("Teacher (5 frame, gallery)", ["ImageBind vision x5", "ImageBind audio"],
         [("ImageBind vision x5", 1), ("ImageBind audio", 1)], None),
        ("MLP student", ["CLIP image", "AST audio"],
         [("CLIP image", 1), ("AST audio", 1)], "MLP head"),
        ("Cross-attention student", ["SigLIP2 image", "CLAP audio"],
         [("SigLIP2 image", 1), ("CLAP audio", 1)], "Cross-attn head"),
        ("Multi-token student", ["SigLIP2 image", "CLAP audio"],
         [("SigLIP2 image", 5), ("CLAP audio", 5)], "Multi-token head"),
    ]

    print("\n=== Per-component (one forward) ===")
    for name, (p, f) in {**enc, **heads}.items():
        print(f"  {name:<24} params {human(p):>10}   FLOPs {human(f):>10}")

    print("\n=== End-to-end pipelines (per clip) ===")
    print(f"{'Pipeline':<28} {'Params':>10} {'FLOPs/clip':>12}")
    print("-" * 54)
    for name, param_comps, flop_comps, head in pipelines:
        # params: unique encoder weights (counted once) + head
        params = sum({c: comp_p(c) for c in param_comps}.values())
        flops = sum(comp_f(c) * mult for c, mult in flop_comps)
        if head:
            params += heads[head][0]
            flops += heads[head][1]
        print(f"{name:<28} {human(params):>10} {human(flops):>12}")

    print(
        "\nNotes:\n"
        " - FLOPs are FlopCounterMode's multiply-add counts for ONE clip; relative, not exact cycles.\n"
        " - Multi-token runs the image encoder x5 (5 frames) and audio encoder x5 (5 windows).\n"
        " - ImageBind params are the whole multimodal model (one checkpoint you load).\n"
        " - Encoder weights are counted once per pipeline even when run multiple times."
    )


if __name__ == "__main__":
    main()
