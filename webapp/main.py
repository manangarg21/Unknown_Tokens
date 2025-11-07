import os
from typing import Dict, Any
from copy import deepcopy

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

from src.utils.io import load_yaml
from src.data.preprocess import TokenizeCollator
from src.models.rcnn import RCNNHead
from src.models.gru import GRUHead
from src.models.lora import attach_lora


class PredictRequest(BaseModel):
    text: str


def build_model(cfg: Dict[str, Any]):
    model_name = cfg["model"].get("alt_backbone") if cfg["model"].get("use_alt_backbone", False) else cfg["model"]["backbone"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)

    if cfg["model"]["lora"]["enabled"]:
        backbone = attach_lora(
            backbone,
            r=cfg["model"]["lora"]["r"],
            alpha=cfg["model"]["lora"]["alpha"],
            dropout=cfg["model"]["lora"]["dropout"],
        )

    hidden_size = backbone.config.hidden_size
    head: nn.Module
    if "rcnn" in cfg["model"]:
        head = RCNNHead(
            hidden_size=hidden_size,
            num_labels=cfg["data"]["num_labels"],
            conv_channels=cfg["model"]["rcnn"]["conv_channels"],
            kernel_sizes=tuple(cfg["model"]["rcnn"]["kernel_sizes"]),
            rnn_hidden=cfg["model"]["rcnn"]["rnn_hidden"],
            rnn_layers=cfg["model"]["rcnn"]["rnn_layers"],
            dropout=cfg["model"]["rcnn"]["dropout"],
        )
    elif "gru" in cfg["model"]:
        head = GRUHead(
            hidden_size=hidden_size,
            num_labels=cfg["data"]["num_labels"],
            hidden=cfg["model"]["gru"]["hidden"],
            layers=cfg["model"]["gru"]["layers"],
            bidirectional=cfg["model"]["gru"].get("bidirectional", True),
            dropout=cfg["model"]["gru"]["dropout"],
        )
    else:
        raise ValueError("Config must contain either model.rcnn or model.gru")

    class M(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, input_ids, attention_mask, **kwargs):
            seq = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return self.head(seq, attention_mask)

    return M(backbone, head), tokenizer


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


app = FastAPI(title="Sarcasm Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state
_device: torch.device = _select_device()
_model: nn.Module = None
_tokenizer: Any = None
_collate: TokenizeCollator | None = None


def _build_and_load(cfg: Dict[str, Any], ckpt: str) -> tuple[nn.Module, Any, TokenizeCollator]:
    def try_load(cand_cfg: Dict[str, Any]) -> tuple[nn.Module, Any, bool]:
        model, tok = build_model(cand_cfg)
        model.to(_device).eval()
        state = torch.load(ckpt, map_location="cpu")
        try:
            model.load_state_dict(state)
            return model, tok, True
        except RuntimeError as e:
            msg = str(e).lower()
            if "base_model.model" in msg or "lora" in msg:
                return model, tok, False
            raise

    # First attempt: as-is
    model, tok, ok = try_load(cfg)
    if not ok:
        # Retry by toggling LoRA flag
        cfg2 = deepcopy(cfg)
        if "lora" in cfg2.get("model", {}):
            cfg2["model"]["lora"]["enabled"] = not bool(cfg2["model"]["lora"]["enabled"])
        model2, tok2, ok2 = try_load(cfg2)
        if ok2:
            model, tok = model2, tok2
        else:
            # Final fallback: non-strict load
            model_fallback, tok_fallback = build_model(cfg)
            model_fallback.to(_device).eval()
            state_fb = torch.load(ckpt, map_location="cpu")
            model_fallback.load_state_dict(state_fb, strict=False)
            model, tok = model_fallback, tok_fallback

    collate = TokenizeCollator(tokenizer=tok, max_length=cfg["model"]["max_length"])
    return model, tok, collate


@app.on_event("startup")
def load_model_on_startup() -> None:
    global _model, _tokenizer, _collate
    # Default config/ckpt (override with env vars if desired)
    cfg_path = os.environ.get(
        "SARCASM_CONFIG",
        os.path.join("configs", "english_roberta_rcnn_sarcasm.yaml"),
    )
    cfg = load_yaml(cfg_path)

    ckpt = os.environ.get(
        "SARCASM_CKPT",
        # prefer a concrete checkpoint if exists, otherwise fall back to config output_dir
        os.path.join("outputs", "english_roberta_rcnn_sarcasm", "best.pt"),
    )
    if not os.path.isfile(ckpt):
        # fallback to config output_dir/best.pt
        alt_ckpt = os.path.join(cfg["output_dir"], "best.pt")
        if os.path.isfile(alt_ckpt):
            ckpt = alt_ckpt
        else:
            raise FileNotFoundError(f"Checkpoint not found at {ckpt} or {alt_ckpt}")

    _model, _tokenizer, _collate = _build_and_load(cfg, ckpt)


@app.get("/", response_class=HTMLResponse)
def serve_index() -> HTMLResponse:
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.isfile(index_path):
        return HTMLResponse(open(index_path, "r", encoding="utf-8").read())
    return HTMLResponse("<h1>Sarcasm Detector</h1><p>Frontend not found.</p>")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="'text' must be a non-empty string")

    text = req.text.strip()
    batch = _collate([{"text": text, "label": 0}])  # dummy label
    input_ids = batch["input_ids"].to(_device)
    attention_mask = batch["attention_mask"].to(_device)

    with torch.no_grad():
        logits = _model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    p_non_sarcastic = float(probs[0])
    p_sarcastic = float(probs[1])
    label = "sarcastic" if p_sarcastic >= p_non_sarcastic else "non-sarcastic"

    return {
        "label": label,
        "probabilities": {
            "non_sarcastic": p_non_sarcastic,
            "sarcastic": p_sarcastic,
        },
    }


@app.get("/static/{file_path:path}")
def serve_static(file_path: str):
    full_path = os.path.join(os.path.dirname(__file__), "static", file_path)
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full_path)


