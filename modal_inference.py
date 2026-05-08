"""
modal_inference.py — TRIBE v2 inference running on Modal GPUs.

The local app calls `predict_video_remote()` instead of running the model locally.
Everything else (FastAPI, reports, brain viz) keeps running on your machine.

SETUP (run once):
    pip install modal
    modal setup                              # authenticate

DOWNLOAD MODELS (run once, ~20-30 min):
    modal run modal_inference.py::download_models

DEPLOY (persistent endpoint):
    modal deploy modal_inference.py

TEST WITHOUT DEPLOYING:
    modal run modal_inference.py::predict_video --video-path /path/to/video.mp4
"""

import modal

# ---------------------------------------------------------------------------
# App & volume
# ---------------------------------------------------------------------------

app = modal.App("tribe-inference")

model_volume = modal.Volume.from_name("tribe-model-cache", create_if_missing=True)
VOLUME_MODEL_CACHE = "/tribe_cache"

# ---------------------------------------------------------------------------
# Container image — only what the model needs, nothing web-related
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy",
        "pandas",
        "huggingface-hub",
        "imageio-ffmpeg",
        "moviepy",
    )
    .pip_install(
        "torch>=2.5.1,<2.7",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "tribev2[plotting] @ https://github.com/facebookresearch/tribev2/archive/72399081ed3f1040c4d996cefb2864a4c46f5b8e.zip"
    )
    # Bake the spaCy model into the image so it doesn't download on every cold start
    .run_commands("python -m spacy download en_core_web_lg")
    # Only copy the two files the model needs — not the full web app
    .add_local_file("tribe_runtime.py", "/app/tribe_runtime.py", copy=True)
    .add_local_file("runtime_setup.py", "/app/runtime_setup.py", copy=True)
    .env({"PYTHONPATH": "/app", "TRIBE_CACHE_DIR": VOLUME_MODEL_CACHE, "IS_MODAL": "1"})
    .workdir("/app")
)

# ---------------------------------------------------------------------------
# One-time model download
# Run: modal run modal_inference.py::download_models
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    volumes={VOLUME_MODEL_CACHE: model_volume},
    timeout=3600,
    env={"TRIBE_CACHE_DIR": VOLUME_MODEL_CACHE},
)
def download_models():
    """Download TRIBE v2 into the persistent volume. Run this once."""
    from huggingface_hub import snapshot_download
    from tribe_runtime import MODEL_REPO, MODEL_SNAPSHOT_DIR, TribeVideoBackend

    print(f"Downloading TRIBE v2 to {MODEL_SNAPSHOT_DIR} …")
    MODEL_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=["config.yaml", "best.ckpt"],
        local_dir=MODEL_SNAPSHOT_DIR,
        local_dir_use_symlinks=False,
    )

    print("Verifying checkpoint loads …")
    TribeVideoBackend().load()

    model_volume.commit()
    print("\nDone! Run `modal deploy modal_inference.py` to go live.")


# ---------------------------------------------------------------------------
# Inference class — model loads once per container, stays in GPU memory
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu="A100",
    volumes={VOLUME_MODEL_CACHE: model_volume},
    timeout=1800,
    scaledown_window=300,
    env={"TRIBE_CACHE_DIR": VOLUME_MODEL_CACHE},
)
class TribeInference:
    @modal.enter()
    def load_model(self):
        import torch
        import transformers
        import tribev2.eventstransforms as et
        import pandas as pd

        # Force HuggingFace models to load in float16 to save VRAM and speed up encoding.
        # But ONLY for video/image models. Audio models (like wav2vec2-bert) often fail
        # with LayerNorm expecting Float32 when forced to Half.
        original_from_pretrained = transformers.PreTrainedModel.from_pretrained
        
        @classmethod
        def fast_from_pretrained(cls, *args, **kwargs):
            model_name = args[0] if args else kwargs.get("pretrained_model_name_or_path", "")
            # Only apply float16 to the massive video/image models
            # We skip audio models like w2v-bert because their LayerNorm fails with Half precision
            if any(name in str(model_name).lower() for name in ["vjepa", "videomae", "llava", "dinov2"]):
                if "torch_dtype" not in kwargs:
                    kwargs["torch_dtype"] = torch.float16
                if "attn_implementation" not in kwargs:
                    kwargs["attn_implementation"] = "sdpa"
            return original_from_pretrained.__func__(cls, *args, **kwargs)
            
        transformers.PreTrainedModel.from_pretrained = fast_from_pretrained

        # Enable TF32 for faster matmuls on Ampere GPUs (like A10G)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Skip word extraction — we drop text events anyway and it wastes 3+ minutes
        def _fast_empty_transcript(wav_filename, language):
            return pd.DataFrame(columns=["text", "start", "duration", "sequence_id", "sentence"])

        et.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(_fast_empty_transcript)

        # ---------------------------------------------------------------------------
        # OPTIONAL: H100 BATCHING PATCH
        # If you ever upgrade the GPU to "H100" and want to cut the video encoding
        # time down by ~1 minute, set this to True. 
        # WARNING: Do NOT set this to True on an A10G or A100, it will crash with OOM!
        # ---------------------------------------------------------------------------
        USE_H100_BATCHING = False
        
        if USE_H100_BATCHING:
            from neuralset.extractors.video import HuggingFaceVideo, _HFVideoModel, _VideoImage
            import neuralset.base as nsbase
            import numpy as np
            from tqdm import tqdm
            import torch.multiprocessing as mp

            original_get_data = HuggingFaceVideo._get_data

            def batched_get_data(self, events):
                if not any(z in self.image.model_name for z in _HFVideoModel.MODELS):
                    yield from original_get_data(self, events)
                    return

                model = _HFVideoModel(
                    model_name=self.image.model_name,
                    pretrained=self.image.pretrained,
                    layer_type=self.layer_type,
                    num_frames=self.num_frames,
                )
                
                # Fix multiprocessing CUDA error: only move to GPU if not already there
                # AND only if we are in the main process (not a DataLoader worker)
                if str(model.model.device) == "cpu" and mp.current_process().name == "MainProcess":
                    model.model.to(self.image.device)

                batch_size = 16 # Force batch size 16 for H100
                
                for event in events:
                    video = event.read()
                    audio = video.audio if self.use_audio else None

                    freq = self.frequency if self.frequency != "native" else event.frequency
                    expect_frames = nsbase.Frequency(freq).to_ind(event.duration)
                    
                    T = 1 / freq if self.clip_duration is None else self.clip_duration
                    subtimes = list(k / model.num_frames * T for k in reversed(range(model.num_frames)))

                    times = np.linspace(0, video.duration, expect_frames + 1)[1:]
                    output = None

                    batch_data = []
                    batch_indices = []

                    for k, t in tqdm(enumerate(times), total=len(times), desc=f"Encoding video (Batch={batch_size})"):
                        ims = [_VideoImage(video=video, time=max(0, t - t2)) for t2 in subtimes]
                        pil_imgs = [i.read() for i in ims]
                        
                        if pil_imgs and self.max_imsize is not None:
                            factor = max(pil_imgs[0].size) / self.max_imsize
                            if factor > 1:
                                size = tuple(int(s / factor) for s in pil_imgs[0].size)
                                pil_imgs = [pi.resize(size) for pi in pil_imgs]
                                
                        data = np.array([np.array(pi) for pi in pil_imgs])
                        batch_data.append(data)
                        batch_indices.append(k)

                        if len(batch_data) == batch_size or k == len(times) - 1:
                            batched_input = np.array(batch_data)
                            t_embd_batch = model.predict_hidden_states(batched_input, None)
                            
                            for i, idx in enumerate(batch_indices):
                                t_embd = t_embd_batch[i]
                                embd = self.image._aggregate_tokens(t_embd).cpu().numpy()
                                if not self.image.cache_all_layers and self.image.cache_n_layers is None:
                                    embd = self.image._aggregate_layers(embd)
                                
                                if output is None:
                                    output = np.zeros((len(times),) + embd.shape)
                                output[idx] = embd
                                
                            batch_data = []
                            batch_indices = []

                    video.close()
                    output = output.transpose(list(range(1, output.ndim)) + [0])
                    yield nsbase.TimedArray(
                        data=output.astype(np.float32),
                        frequency=freq,
                        start=nsbase._UNSET_START,
                        duration=event.duration,
                    )

            HuggingFaceVideo._get_data = batched_get_data
            print("H100 Batching Patch Applied!")

        from tribe_runtime import TribeVideoBackend
        self.backend = TribeVideoBackend()
        self.backend.load()
        print("Model loaded and ready.")

    @modal.method()
    def predict_video(self, video_bytes: bytes, filename: str = "video.mp4") -> dict:
        """
        Run TRIBE v2 inference on a video and return a JSON-serializable result dict.
        Called by the local tribe_runtime.py when TRIBE_USE_MODAL=1 is set.
        """
        import os
        import tempfile
        import numpy as np
        import torch

        with tempfile.NamedTemporaryFile(suffix=f"_{filename}", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        # inference_mode disables gradient tracking — frees VRAM and speeds up compute
        with torch.inference_mode():
            result = self.backend.predict_video(tmp_path)

        os.unlink(tmp_path)

        preds_np = np.asarray(result.preds)
        return {
            "preds": preds_np.tolist(),
            "preds_shape": list(preds_np.shape),
            "preds_dtype": str(preds_np.dtype),
            "timestamps": result.timestamps,
            "device": result.device,
            "modalities": result.modalities,
            "events_count": result.events_count,
            "mesh_level": result.mesh_level,
        "subject_model": result.subject_model,
        "hemodynamic_lag_seconds": result.hemodynamic_lag_seconds,
    }


# ---------------------------------------------------------------------------
# Manual test: modal run modal_inference.py::predict_video --video-path ...
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(video_path: str = ""):
    if not video_path:
        print("Usage: modal run modal_inference.py --video-path /path/to/video.mp4")
        return
    video_bytes = open(video_path, "rb").read()
    print(f"Sending {len(video_bytes) / 1e6:.1f} MB to Modal …")
    instance = TribeInference()
    result = instance.predict_video.remote(video_bytes, filename=video_path.split("/")[-1])
    import numpy as np
    preds = np.array(result["preds"])
    print(f"Result: preds shape={preds.shape}, timestamps={len(result['timestamps'])}, device={result['device']}")
