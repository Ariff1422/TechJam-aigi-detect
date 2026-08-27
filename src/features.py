"""CLIP/DINOv2 loading + embedding extraction. PLAN.md Section 1.1/1.1b/1.2/3.3."""
import torch
import open_clip
import timm
from PIL import Image

_MODEL_CACHE = {}

_CLIP_NAME_MAP = {
    # open_clip's plain "ViT-B-32"/"ViT-L-14" configs use quick_gelu=False, but the
    # OpenAI-pretrained checkpoints were trained with QuickGELU — use the "-quickgelu"
    # variant with pretrained="openai" to avoid an activation-function mismatch.
    "clip_vitb32": "ViT-B-32-quickgelu",
    "clip_vitl14": "ViT-L-14-quickgelu",
}

# timm model names for the non-CLIP backbones (Section 1.1b/7 comparison).
# dinov2_vitl14 ~300M params, comfortably under the 2B cap (Section 1.3).
_TIMM_NAME_MAP = {
    "dinov2_vitl14": "vit_large_patch14_dinov2",
}


class _TimmEncoderAdapter:
    """Wraps a timm model so it exposes an .encode_image() method, matching
    open_clip's interface — lets features.py/robustness_eval.py/train.py
    call the same method regardless of which backbone family is loaded."""

    def __init__(self, model):
        self.model = model

    def encode_image(self, x):
        return self.model(x)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()


def load_backbone(family: str = "clip_vitb32", pretrained: str = "openai", device: str = "cpu"):
    """Load a frozen vision encoder (CLIP or DINOv2) + its preprocessing transform.

    Returns (model, preprocess, embedding_dim). embedding_dim is read from
    the model's actual output shape, not hardcoded (PLAN.md Section 1.2).
    model exposes .encode_image() regardless of backbone family.
    """
    key = (family, pretrained, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if family in _CLIP_NAME_MAP:
        open_clip_name = _CLIP_NAME_MAP[family] if pretrained == "openai" else family
        model, _, preprocess = open_clip.create_model_and_transforms(
            open_clip_name, pretrained=pretrained
        )
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        input_size = 224

    elif family in _TIMM_NAME_MAP:
        timm_model = timm.create_model(_TIMM_NAME_MAP[family], pretrained=True, num_classes=0)
        timm_model = timm_model.to(device).eval()
        for p in timm_model.parameters():
            p.requires_grad = False
        cfg = timm.data.resolve_data_config({}, model=timm_model)
        preprocess = timm.data.create_transform(**cfg)
        input_size = cfg["input_size"][-1]
        model = _TimmEncoderAdapter(timm_model)

    else:
        raise ValueError(f"Unsupported backbone family: {family}")

    # Verify embedding dim with an actual shape check rather than trusting config.
    with torch.no_grad():
        dummy = torch.zeros(1, 3, input_size, input_size, device=device)
        emb = model.encode_image(dummy)
    embedding_dim = emb.shape[-1]

    _MODEL_CACHE[key] = (model, preprocess, embedding_dim)
    return model, preprocess, embedding_dim


@torch.no_grad()
def extract_embeddings_from_images(images, model, preprocess, device: str = "cpu", batch_size: int = 32):
    """Extract L2-normalized CLIP embeddings for a list of in-memory PIL images.

    Shared by extract_embeddings (loads from disk) and robustness_eval.py
    (applies a transform in memory first, per PLAN.md Phase 2), so
    transformed images never need to be written back to disk.
    Returns a (N, embedding_dim) torch.Tensor on CPU.
    """
    all_embeddings = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        batch = torch.stack([preprocess(img.convert("RGB")) for img in batch_imgs]).to(device)
        emb = model.encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings, dim=0)


@torch.no_grad()
def extract_embeddings(image_paths, model, preprocess, device: str = "cpu", batch_size: int = 32):
    """Extract L2-normalized CLIP embeddings for a list of image paths.

    Returns a (N, embedding_dim) torch.Tensor on CPU.
    """
    all_embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        all_embeddings.append(extract_embeddings_from_images(imgs, model, preprocess, device, batch_size))
    return torch.cat(all_embeddings, dim=0)
