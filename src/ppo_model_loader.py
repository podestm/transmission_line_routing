"""Model loading helpers for PPO routing."""

import io
import os
import tempfile
import zipfile

import torch
from sb3_contrib import MaskablePPO


# umi nacist i starsi checkpointy, ktere maji torch compile prefixy v klicich
def load_ppo_model(model_path):
    """Load MaskablePPO, transparently fixing torch.compile() _orig_mod key prefix."""
    model_path = str(model_path)

    try:
        return MaskablePPO.load(model_path)
    except RuntimeError as exc:
        if "_orig_mod" not in str(exc):
            raise
        print("  Detected torch.compile() artifact - remapping _orig_mod keys...")

    zip_path = model_path if model_path.endswith(".zip") else model_path + ".zip"

    def _remap(obj):
        if isinstance(obj, dict):
            return {k.replace("._orig_mod.", "."): _remap(v) for k, v in obj.items()}
        return obj

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    os.close(tmp_fd)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf_in, \
             zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf_out:
            for name in zf_in.namelist():
                data = zf_in.read(name)
                if name.endswith(".pth"):
                    try:
                        state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
                        state = _remap(state)
                        buf = io.BytesIO()
                        torch.save(state, buf)
                        data = buf.getvalue()
                    except Exception:
                        pass
                zf_out.writestr(name, data)
        return MaskablePPO.load(tmp_path)
    finally:
        os.unlink(tmp_path)
