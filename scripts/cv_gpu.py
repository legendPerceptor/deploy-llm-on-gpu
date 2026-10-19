"""Reuse oGMemory's face API, fail startup if any model loses CUDA."""


def validate_sessions(models):
    if not {"detection", "recognition"}.issubset(models):
        raise RuntimeError("Detection and recognition models are both required")
    providers = {}
    for name, model in models.items():
        active = model.session.get_providers()
        if not active or active[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"{name}: CUDA is not the primary provider: {active}")
        providers[name] = active
    return providers


def main():
    # Import torch first so its CUDA/cuDNN libraries are loaded before ORT.
    import torch
    import onnxruntime as ort
    import uvicorn
    from fastapi import HTTPException
    from cv_service import InsightFaceBackend, build_app

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CUDA provider is unavailable")
    backend = InsightFaceBackend(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    providers = validate_sessions(backend._app.models)
    app = build_app(backend)

    @app.get("/health/gpu")
    def health_gpu():
        return {
            "ok": True, "model": backend.model_name, "embedding_dimension": 512,
            "device": torch.cuda.get_device_name(0), "providers": providers,
        }

    # Do not let placeholder routes masquerade as completed CV inference.
    app.router.routes[:] = [
        route for route in app.router.routes
        if getattr(route, "path", None) not in {"/detect/objects", "/detect/pose"}
    ]

    @app.post("/detect/objects")
    @app.post("/detect/pose")
    def unsupported():
        raise HTTPException(501, "This deployment implements face detection/embeddings only")

    uvicorn.run(app, host="0.0.0.0", port=8081, workers=1)


if __name__ == "__main__":
    main()
