"""Helpers shared by baseline generation backends."""


def normalize_generation_backend(value):
    """Normalize backend config to a supported backend name."""
    backend = str(value or "custom").strip().lower().replace("-", "_")
    if backend not in {"custom", "transformers", "vllm"}:
        raise ValueError(
            f"Unsupported generation_backend: {value}. "
            f"Expected one of custom/transformers/vllm."
        )
    return backend


def is_greedy_temperature(temperature):
    """Return whether a temperature should use greedy decoding."""
    return float(temperature or 0.0) < 1e-5


def import_vllm():
    """Import vLLM lazily so non-vLLM runs keep the dependency optional."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "The vllm backend was requested, but `vllm` is not installed in the current environment."
        ) from exc
    return LLM, SamplingParams


def build_vllm_sampling_params(max_new_tokens, temperature, stop_token_ids=None):
    """Create a vLLM sampling-parameter object for baseline generation."""
    _, SamplingParams = import_vllm()
    kwargs = {
        "max_tokens": max(int(max_new_tokens), 0),
    }
    if stop_token_ids:
        kwargs["stop_token_ids"] = [int(token_id) for token_id in stop_token_ids]
    if is_greedy_temperature(temperature):
        kwargs["temperature"] = 0.0
    else:
        kwargs["temperature"] = float(temperature)
    return SamplingParams(**kwargs)


def extract_vllm_timing_ms(request_output, fallback_total_ms):
    """Extract TTFT and total wall time from a vLLM request output when available."""
    metrics = getattr(request_output, "metrics", None)
    if metrics is None:
        return fallback_total_ms, fallback_total_ms

    arrival_time = getattr(metrics, "arrival_time", None)
    first_token_time = getattr(metrics, "first_token_time", None)
    finished_time = getattr(metrics, "finished_time", None)
    last_token_time = getattr(metrics, "last_token_time", None)

    if finished_time is None:
        finished_time = last_token_time

    ttft_ms = fallback_total_ms
    total_ms = fallback_total_ms
    if arrival_time is not None and first_token_time is not None:
        ttft_ms = max((first_token_time - arrival_time) * 1000.0, 0.0)
    if arrival_time is not None and finished_time is not None:
        total_ms = max((finished_time - arrival_time) * 1000.0, 0.0)
    return total_ms, ttft_ms
