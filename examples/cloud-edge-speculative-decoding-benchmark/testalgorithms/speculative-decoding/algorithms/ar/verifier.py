"""
Verify-side runtime for the cloud-edge speculative decoding benchmark.
"""

import math
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPECDEC_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
for path in (MODULE_DIR, SPECDEC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from base_verifier import BaseSpeculativeVerifier
from common.config_utils import _to_bool, _to_int, _to_optional_int
from common.generation_backends import (
    build_vllm_sampling_params,
    extract_vllm_timing_ms,
    import_vllm,
    is_greedy_temperature,
    normalize_generation_backend,
)
from common.request_utils import build_single_path_response, compute_perf, normalize_request
from common.stop_utils import apply_stop_to_sequence, is_stop_token
from result_builder import record_sample_output
from sedna.common.class_factory import ClassFactory, ClassType
from common.timeline_utils import now_ns, offset_ns

os.environ["BACKEND_TYPE"] = "TORCH"

def _sample_token_tensor(logits, temperature=0.0):

    """
    Sample one token id from logits with temperature scaling.

    Parameters
    ----------
    logits : torch.Tensor
        Usually shape `[1, vocab]`.
    temperature : float, default=0.0
        Shared sampling temperature used by both draft and verify sides.

    Returns
    -------
    torch.Tensor
        Shape `[1, 1]`, containing the sampled token id.
    """
    if float(temperature or 0.0) < 1e-5:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled_temperature = max(float(temperature or 0.0), 1e-5)
    scaled_logits = logits.float() / scaled_temperature
    return torch.distributions.Categorical(logits=scaled_logits).sample().unsqueeze(-1)

def _scaled_logits(logits, temperature=0.0):
    """
    Apply temperature scaling to raw logits.

    Returns
    -------
    torch.Tensor
        Float tensor with the same semantic shape as `logits`.
    """
    scaled_temperature = max(float(temperature or 0.0), 1e-5)
    return logits.float() / scaled_temperature

def _probs_from_logits(logits, temperature=0.0):

    """
    Convert logits into probability space.

    Returns
    -------
    torch.Tensor
        Probability tensor over the vocabulary.
    """
    return torch.softmax(_scaled_logits(logits, temperature=temperature), dim=-1)

def _sample_from_probs(probs):
    """
    Sample a token id from a normalized probability vector.

    Parameters
    ----------
    probs : torch.Tensor
        One-dimensional probability vector.

    Returns
    -------
    int
        Sampled token id.
    """
    return int(torch.distributions.Categorical(probs=probs).sample().item())

def _sample_from_residual(p_probs, q_probs):

    """
    Sample a correction token from residual mass `max(p-q, 0)`.

    This is the strict speculative-decoding rejection rule.

    Parameters
    ----------
    p_probs : torch.Tensor
        Verifier probability vector.
    q_probs : torch.Tensor
        Drafter probability vector.

    Returns
    -------
    int
        Correction token id. Falls back to sampling from `p_probs` if the
        residual distribution is empty.
    """
    residual = torch.clamp(p_probs - q_probs, min=0.0)
    if float(residual.sum().item()) <= 0.0:
        return _sample_from_probs(p_probs)
    residual = residual / residual.sum()
    return _sample_from_probs(residual)

def _greedy_token_id_from_logits(logits):
    """Return the deterministic argmax token id for one logits tensor."""
    return int(torch.argmax(logits, dim=-1).reshape(-1)[0].item())

def _align_probability_vectors(p_probs, q_probs):
    """
    Align verifier and drafter probability vectors to the same vocabulary size.

    Parameters
    ----------
    p_probs : torch.Tensor
        Verifier probabilities.
    q_probs : torch.Tensor
        Drafter probabilities.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Length-aligned `(p_probs, q_probs)`.
    """
    # Current experiments may use draft / verify models whose vocab sizes are not
    # exactly equal. This keeps the strict path runnable by zero-padding the shorter
    # side, but it is still worth treating mismatched vocab as a first-class suspect
    # when acceptance looks abnormal.
    p_probs = p_probs.reshape(-1)
    q_probs = q_probs.reshape(-1)
    if p_probs.shape[0] == q_probs.shape[0]:
        return p_probs, q_probs
    if p_probs.shape[0] > q_probs.shape[0]:
        padding = torch.zeros(
            p_probs.shape[0] - q_probs.shape[0],
            dtype=q_probs.dtype,
            device=q_probs.device,
        )
        q_probs = torch.cat([q_probs, padding], dim=0)
        return p_probs, q_probs
    padding = torch.zeros(
        q_probs.shape[0] - p_probs.shape[0],
        dtype=p_probs.dtype,
        device=p_probs.device,
    )
    p_probs = torch.cat([p_probs, padding], dim=0)
    return p_probs, q_probs

def _ensure_dynamic_cache(past_key_values):
    """Normalize empty or tuple-style cache values into a DynamicCache object."""
    if past_key_values is None:
        return DynamicCache()
    if isinstance(past_key_values, DynamicCache):
        return past_key_values
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(past_key_values)
    return past_key_values

def _cache_seq_length(past_key_values):
    """Return the current token length already materialized inside a cache object."""
    cache = _ensure_dynamic_cache(past_key_values)
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    return 0

@ClassFactory.register(ClassType.GENERAL, alias="SpeculativeVerifyModel")
class SpeculativeVerifyModel(BaseSpeculativeVerifier):
    """
    Verify runtime for speculative decoding.

    Responsibilities:
    - load and own the verifier model
    - provide cloud-only generation
    - maintain verifier-side KV cache across speculative rounds
    - perform strict acceptance / correction for collaboration rounds
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.kwargs = kwargs

        self.algorithm_name = "ar_spec"
        self.model_name = kwargs.get("model", "Qwen/Qwen2.5-7B-Instruct")

        configured_device = kwargs.get("device", "auto")
        if configured_device == "auto":
            configured_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = configured_device
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.trust_remote_code = _to_bool(kwargs.get("trust_remote_code", True), True)

        self.default_prompt_tokens = _to_optional_int(kwargs.get("prompt_tokens"))
        self.default_completion_tokens = _to_optional_int(kwargs.get("max_new_tokens"))
        self.runaway_guard_max_new_tokens = _to_optional_int(
            kwargs.get("runaway_guard_max_new_tokens"),
            512,
        )
        self.draft_tokens_per_step = max(1, _to_int(kwargs.get("draft_tokens_per_step"), 8))
        self.sample_temperature = float(kwargs.get("sample_temperature", 0.0))
        self.generation_backend = normalize_generation_backend(
            kwargs.get("generation_backend", "custom")
        )
        self.stop_mode = str(kwargs.get("stop_mode", "choice")).strip().lower().replace("-", "_")

        self.enable_network_sleep = _to_bool(kwargs.get("enable_network_sleep", False), False)
        self.network_rtt_ms = max(0.0, float(kwargs.get("network_rtt_ms", 0.0)))
        self.network_jitter_ms = max(0.0, float(kwargs.get("network_jitter_ms", 0.0)))
        self.network_uplink_ratio = min(
            1.0,
            max(0.0, float(kwargs.get("network_uplink_ratio", 0.5))),
        )
        self.network_uplink_bandwidth_mbps = max(
            0.0,
            float(kwargs.get("network_uplink_bandwidth_mbps", 0.0)),
        )
        self.network_downlink_bandwidth_mbps = max(
            0.0,
            float(kwargs.get("network_downlink_bandwidth_mbps", 0.0)),
        )
        self.network_seed = kwargs.get("network_seed")
        self.network_rng = (
            random.Random(int(self.network_seed))
            if self.network_seed is not None
            else random.Random()
        )
        self.sample_output_log = kwargs.get("sample_output_log")

        self.tokenizer = None
        self.model = None
        self.vllm_model = None

        self._verify_sessions = {}

    def start_session(
        self,
        data=None,
        request=None,
        draft_session=None,
        prompt_ids=None,
        prompt_token_count=None,
        **kwargs
    ):
        """
        Create verifier-side session state for one request.

        Parameters
        ----------
        data : dict | None
            Raw dataset item.
        request : dict | None
            Optional pre-normalized request.
        draft_session : dict | None
            Drafter session, used to reuse prompt ids in collaboration mode.
        prompt_ids : torch.Tensor | None
            Optional pre-tokenized prompt.
        prompt_token_count : int | None
            Optional prompt length override.

        Returns
        -------
        dict
            Verifier session with keys:
            `request_id`, `request`, `prompt_ids`, `prompt_token_count`,
            `committed_ids`, `_draft_session`.
        """
        # The verifier session is intentionally lightweight: it stores request,
        # prompt ids, and the committed token prefix that has been accepted so far.
        # Actual KV materialization lives in `_verify_sessions`.
        self._ensure_loaded(require_torch_model=self.generation_backend != "vllm")
        if request is None:
            request = normalize_request(
                data,
                self.default_prompt_tokens,
                self.default_completion_tokens,
                _to_optional_int,
            )
        if draft_session is not None:

            prompt_ids = prompt_ids or draft_session.get("prompt_ids")
            prompt_token_count = prompt_token_count or draft_session.get("prompt_token_count")
        if prompt_ids is None:
            prompt_ids = self._prepare_prompt_ids(request)
        if prompt_token_count is None:
            prompt_token_count = int(prompt_ids.shape[1])
        request = dict(request)
        request["completion_tokens"] = self._resolve_completion_limit(
            prompt_token_count,
            request.get("completion_tokens"),
        )
        return {
            "request_id": str(request.get("request_id", "default")),
            "request": request,
            "prompt_ids": prompt_ids,
            "prompt_token_count": int(prompt_token_count),
            "committed_ids": [],
            "_draft_session": draft_session,
        }

    def verify(self, session, draft_output=None, draft_ids=None, **kwargs):
        """
        Verify one drafter round and produce feedback for the next step.

        Parameters
        ----------
        session : dict
            Verifier session from `start_session`.
        draft_output : dict | None
            Drafter payload. Expected keys include `draft_ids` and `draft_logits`.
        draft_ids : list[int] | None
            Optional explicit draft ids override.

        Returns
        -------
        dict
            Verifier payload with accepted/corrected tokens and a `feedback`
            field shaped as:
            `{"draft_output": dict, "verify_output": dict}`.
        """
        # The Sedna paradigm hands draft output into the verifier. The example
        # runtime translates that abstract payload into strict speculative checks.
        if draft_ids is None:
            draft_ids = list((draft_output or {}).get("draft_ids", []))
        draft_session = session.get("_draft_session")
        draft_state = (draft_session or {}).get("state", {}) if isinstance(draft_session, dict) else {}
        current_round = int(draft_state.get("rounds", 0)) + 1
        payload = self.verify_tokens(
            request=session["request"],
            prompt_ids=session["prompt_ids"],
            committed_ids=session["committed_ids"],
            draft_ids=draft_ids,
            draft_logits=(draft_output or {}).get("draft_logits"),
            step_index=current_round,
        )

        committed_ids = (
            list(session["committed_ids"])
            + list(payload.get("accepted_ids", []))
            + list(payload.get("corrected_ids", []))
        )
        session["committed_ids"] = committed_ids
        commit_payload = self.append_committed(
            session,
            committed_ids=committed_ids,
            accepted_ids=list(payload.get("accepted_ids", [])),
            corrected_ids=list(payload.get("corrected_ids", [])),
            decision_state=payload.get("decision_state"),
        )
        payload["cloud_compute_ms"] = float(payload.get("cloud_compute_ms", 0.0) or 0.0) + float(
            commit_payload.get("cloud_compute_ms", 0.0) or 0.0
        )
        control = {
            "stop": bool(payload.get("stop", False)),
            "progress": bool(
                payload.get("accepted_ids")
                or payload.get("corrected_ids")
                or draft_ids
                or payload.get("stop", False)
            ),
            "result": None,
        }
        payload["stop"] = bool(control.get("stop", False))
        payload["progress"] = bool(control.get("progress", True))
        if control.get("result") is not None:
            payload["result"] = control.get("result")
        payload["feedback"] = {
            "draft_output": dict(draft_output or {}),
            "verify_output": {
                key: value
                for key, value in payload.items()
                if key not in {"feedback"}
            },
        }
        return payload

    def close_session(self, session, request=None):
        """
        Close verifier request scope and release request-local cache.

        Parameters
        ----------
        session : dict
            Verifier session.
        request : dict | None
            Optional request override. Defaults to `session["request"]`.

        Returns
        -------
        None
        """
        request = request or session.get("request")
        self.finalize_request(request=request)

    @torch.no_grad()
    def inference(self, data, token_callback=None, **kwargs):
        """
        Run verifier as a standalone cloud-only generator.

        Parameters
        ----------
        data : dict
            Raw / normalized dataset sample.
        token_callback : Callable | None
            Optional token streaming callback.

        Returns
        -------
        dict
            Ianvs response object for cloud-only inference.
        """
        self._ensure_loaded(require_torch_model=self.generation_backend != "vllm")
        # Cloud-only path is intentionally simpler than collaboration:
        # no draft/verify loop, just one normal autoregressive generation path
        # plus optional benchmark-side network delay simulation.

        request = normalize_request(
            data,
            self.default_prompt_tokens,
            self.default_completion_tokens,
            _to_optional_int,
        )
        prompt_ids = self._prepare_prompt_ids(request)
        prompt_token_count = int(prompt_ids.shape[1])
        request["completion_tokens"] = self._resolve_completion_limit(
            prompt_token_count,
            request.get("completion_tokens"),
        )
        uplink_bytes = self._estimate_cloud_only_uplink_bytes(request, prompt_ids)
        network = self._sample_network_delay(
            uplink_bytes=uplink_bytes,
            downlink_bytes=0,
        )
        request_start_abs_ns = now_ns()
        if self.enable_network_sleep and network["uplink_ms"] > 0.0:
            time.sleep(network["uplink_ms"] / 1000.0)

        completion_ids, compute_ms, ttft_ms, generation_timestamps = self._generate(
            prompt_ids,
            request["completion_tokens"],
            token_callback=token_callback,
            request=request,
        )
        downlink_bytes = self._estimate_cloud_only_downlink_bytes(completion_ids)
        network["downlink_bytes"] = int(downlink_bytes)
        network["downlink_transfer_ms"] = self._bandwidth_delay_ms(
            downlink_bytes,
            self.network_downlink_bandwidth_mbps,
        )
        network["downlink_ms"] = float(network.get("downlink_base_ms", 0.0)) + float(
            network["downlink_transfer_ms"]
        )
        network["network_ms"] = float(network.get("uplink_ms", 0.0)) + float(network["downlink_ms"])
        network["network_half_ms"] = network["network_ms"] / 2.0
        if self.enable_network_sleep and network["downlink_ms"] > 0.0:
            time.sleep(network["downlink_ms"] / 1000.0)
        request_end_abs_ns = now_ns()
        total_ms = compute_ms + network["network_ms"]
        ttft_total_ms = ttft_ms + network["uplink_ms"]
        perf = compute_perf(
            total_latency_ms=total_ms,
            completion_tokens=len(completion_ids),
            ttft_ms=ttft_total_ms,
        )
        response = build_single_path_response(
            prompt_tokens=prompt_token_count,
            completion_tokens=len(completion_ids),
            completion_text=self.tokenizer.decode(completion_ids, skip_special_tokens=True),
            perf=perf,
            simulation={
                "mode": "cloud-only",
                "routed_to": "cloud",
                "acceptance_rate": "",
                "end_to_end_latency": round(total_ms / 1000.0, 6),
                "rounds": 1,
                "accepted_draft_tokens": 0,
                "corrected_tokens": 0,
                "total_draft_tokens": 0,
                "network_overhead_ms": round(network["network_ms"], 6),
                "network_rtt_ms": round(network["network_rtt_ms"], 6),
                "network_jitter_ms": round(network["network_jitter_ms"], 6),
                "network_rtt_base_ms": round(network.get("network_rtt_base_ms", network["network_rtt_ms"]), 6),
                "uplink_base_ms": round(network.get("uplink_base_ms", network["uplink_ms"]), 6),
                "downlink_base_ms": round(network.get("downlink_base_ms", network["downlink_ms"]), 6),
                "uplink_bytes": int(network.get("uplink_bytes", 0)),
                "downlink_bytes": int(network.get("downlink_bytes", 0)),
                "uplink_transfer_ms": round(network.get("uplink_transfer_ms", 0.0), 6),
                "downlink_transfer_ms": round(network.get("downlink_transfer_ms", 0.0), 6),
                "cloud_compute_ms": round(compute_ms, 6),
                "draft_tokens_per_step": self.draft_tokens_per_step,
                "generation_backend": self.generation_backend,
                "task_name": request.get("task_name", "default"),
                "stop_reason": generation_timestamps.get("stop_reason", ""),
            },
            timestamps={
                "request_end_ns": offset_ns(request_start_abs_ns, request_end_abs_ns),
                "first_token_ns": offset_ns(
                    request_start_abs_ns,
                    generation_timestamps["first_token_ns"] or request_end_abs_ns,
                ),
            },
        )
        record_sample_output(
            self,
            {

                "record_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "request_id": request.get("request_id"),
                "mode": "cloud-only",
                "task_name": request.get("task_name", "default"),
                "prompt": request.get("query", ""),
                "gold": request.get("gold", ""),
                "completion": response["completion"],
                "stop_reason": generation_timestamps.get("stop_reason", ""),
            },
        )
        return response

    def finalize_request(self, request=None, request_id=None):
        """
        Release one verifier request session from `_verify_sessions`.

        Parameters
        ----------
        request : dict | None
            Request carrying `request_id`.
        request_id : str | None
            Explicit override.

        Returns
        -------
        None
        """

        rid = request_id
        if rid is None and isinstance(request, dict):
            rid = request.get("request_id")
        if rid is not None:
            self._verify_sessions.pop(str(rid), None)

    def cleanup(self):
        """
        Release all verifier-owned runtime resources.

        Returns
        -------
        None
        """

        self.model = None
        self.vllm_model = None
        self.tokenizer = None
        self._verify_sessions.clear()
    def verify_tokens(
        self,
        request,
        prompt_ids,
        committed_ids,
        draft_ids,
        draft_logits=None,
        step_index=0,
    ):
        """
        Execute the strict speculative-decoding acceptance algorithm.

        This is the most important function for diagnosing low acceptance rate.
        The high-level flow is:
        1. sync verifier cache to trusted prefix
        2. materialize pending logical suffix + new draft block in one forward
        3. compare p/q token-by-token
        4. accept prefix, then emit correction or bonus token
        5. attach timing / network / round stats for later analysis

        Parameters
        ----------
        request : dict
            Normalized request with at least `request_id`, `query`,
            `completion_tokens`.
        prompt_ids : torch.Tensor
            Shape `[1, prompt_len]`.
        committed_ids : list[int]
            Logical committed completion prefix before this round.
        draft_ids : list[int]
            Fresh draft block proposed by drafter.
        draft_logits : list[torch.Tensor] | torch.Tensor | None
            Drafter q logits aligned one-to-one with `draft_ids`.
        step_index : int, default=0
            Human-readable round index for debugging output.

        Returns
        -------
        dict
            Verifier round payload with structure:
            - `accepted_ids`: list[int]
            - `corrected_ids`: list[int]
            - `rejected_draft_ids`: list[int]
            - `decision_state`: dict used by `append_committed`
            - `stop`: bool
            - `stop_reason`: str
            - `cloud_compute_ms`: float
            - `network_overhead_ms`: float
            - `round_stats`: dict[str, int | bool | str]
        """
        self._ensure_loaded()
        # One strict speculative round does four conceptual things:
        # 1) read the verifier cache at the trusted materialized prefix
        # 2) prepend any lazily pending committed suffix before the fresh draft block
        # 3) accept / reject only the fresh draft tokens using p and q
        # 4) materialize only the old pending suffix + accepted draft prefix
        _ = normalize_request(
            request,
            self.default_prompt_tokens,
            self.default_completion_tokens,
            _to_optional_int,
        )

        verify_start = time.perf_counter()
        accepted_ids = []
        corrected_ids = []
        stop = False
        stop_reason = ""
        uplink_bytes = self._estimate_collaboration_uplink_bytes(
            request=request,
            draft_ids=draft_ids,
            draft_logits=draft_logits,
        )
        network = self._sample_network_delay(
            uplink_bytes=uplink_bytes,
            downlink_bytes=0,
        )
        if self.enable_network_sleep and network["uplink_ms"] > 0.0:
            time.sleep(network["uplink_ms"] / 1000.0)

        verify_outputs = None
        pending_materialization_count = 0
        completion_limit = max(int(request.get("completion_tokens", 1)), 1)
        greedy_mode = float(self.sample_temperature or 0.0) < 1e-5
        if draft_ids:
            # `draft_logits` is the drafter-side q distribution for each proposed
            # token. If this list is missing or misaligned, strict acceptance math
            # becomes invalid immediately.
            q_logits_batch = None
            if not greedy_mode:
                if draft_logits is None:
                    raise RuntimeError(
                        f"Strict speculative sampling requires per-step draft logits; "
                        f"got 0 for {len(draft_ids)} draft ids."
                    )
                if isinstance(draft_logits, torch.Tensor):
                    if draft_logits.ndim == 1:
                        draft_logits = draft_logits.unsqueeze(0)
                    if draft_logits.shape[0] != len(draft_ids):
                        raise RuntimeError(
                            f"Strict speculative sampling requires per-step draft logits; "
                            f"got {draft_logits.shape[0]} for {len(draft_ids)} draft ids."
                        )
                    q_logits_batch = draft_logits
                else:
                    if not draft_logits or len(draft_logits) != len(draft_ids):
                        raise RuntimeError(
                            f"Strict speculative sampling requires per-step draft logits; "
                            f"got {0 if not draft_logits else len(draft_logits)} for {len(draft_ids)} draft ids."
                        )
                    q_logits_batch = torch.cat(
                        [
                            item if item.ndim == 2 else item.unsqueeze(0)
                            for item in draft_logits
                        ],
                        dim=0,
                    )
            prefix_ids = self._build_prefix(
                prompt_ids.to(self.device),
                list(committed_ids) + list(draft_ids),
            )
            with torch.no_grad():
                verify_outputs = self.model(
                    input_ids=prefix_ids,
                    use_cache=True,
                )
            prefix_length = int(prompt_ids.shape[1]) + int(len(committed_ids))
            p_logits_for_verify_inputs = [
                verify_outputs.logits[:, prefix_length - 1 + idx, :]
                for idx in range(len(draft_ids))
            ]
            current_next_logits = verify_outputs.logits[:, -1, :]
            rejection_index = None
            logical_prefix_len = len(committed_ids)
            next_logits_after_accept = (
                p_logits_for_verify_inputs[0]
                if p_logits_for_verify_inputs
                else current_next_logits
            )
            for idx, token_id in enumerate(draft_ids):
                # p = verifier distribution, q = drafter distribution.
                # Acceptance uses min(1, p(x) / q(x)) from strict speculative sampling.
                token_id = int(token_id)
                p_logits = p_logits_for_verify_inputs[idx]
                if greedy_mode:
                    verifier_token = _greedy_token_id_from_logits(p_logits)
                    if verifier_token == token_id:
                        accepted_ids.append(token_id)
                        if idx + 1 < len(p_logits_for_verify_inputs):
                            next_logits_after_accept = p_logits_for_verify_inputs[idx + 1]
                        else:
                            next_logits_after_accept = current_next_logits
                        if logical_prefix_len + len(accepted_ids) >= completion_limit:
                            stop_reason = "completion_limit"
                            stop = True
                            rejection_index = idx
                            corrected_ids = []
                            break
                        continue
                    rejection_index = idx
                    corrected_ids = [verifier_token]
                    stop = self._is_stop_token(corrected_ids[-1])
                    if stop:
                        stop_reason = "eos"
                    break
                p_probs = _probs_from_logits(
                    p_logits,
                    temperature=self.sample_temperature,
                ).reshape(-1)
                q_logits = q_logits_batch[idx]
                if q_logits.device != p_probs.device:
                    q_logits = q_logits.to(p_probs.device)
                q_probs = torch.softmax(q_logits, dim=-1).reshape(-1)
                p_probs, q_probs = _align_probability_vectors(p_probs, q_probs)
                if token_id >= p_probs.shape[0] or token_id >= q_probs.shape[0]:
                    raise RuntimeError(
                        f"Draft token id {token_id} exceeds aligned vocab sizes "
                        f"p={p_probs.shape[0]}, q={q_probs.shape[0]}."
                    )
                p_token = float(p_probs[token_id].item())
                q_token = max(float(q_probs[token_id].item()), 1e-12)
                accept_prob = min(1.0, p_token / q_token)

                if torch.rand(1, device=p_probs.device).item() <= accept_prob:
                    accepted_ids.append(token_id)
                    if idx + 1 < len(p_logits_for_verify_inputs):
                        next_logits_after_accept = p_logits_for_verify_inputs[idx + 1]
                    else:
                        next_logits_after_accept = current_next_logits
                    if logical_prefix_len + len(accepted_ids) >= completion_limit:
                        stop_reason = "completion_limit"
                        stop = True
                        rejection_index = idx
                        corrected_ids = []
                        break
                    continue
                rejection_index = idx
                # On rejection, sample one correction token from the residual
                # distribution instead of simply taking argmax from the verifier.
                corrected_ids = [_sample_from_residual(p_probs, q_probs)]
                stop = self._is_stop_token(corrected_ids[-1])
                if stop:
                    stop_reason = "eos"
                break

            if rejection_index is None:

                accepted_ids = list(draft_ids)
                next_logits_after_accept = current_next_logits
                if accepted_ids and self._is_stop_token(accepted_ids[-1]):
                    corrected_ids = []
                    stop_reason = "eos"
                    stop = True
                elif logical_prefix_len + len(accepted_ids) >= completion_limit:
                    corrected_ids = []
                    stop_reason = "completion_limit"
                    stop = True
                else:
                    # If the whole draft block is accepted, strict SD samples one extra
                    # token from p_{gamma+1} so the verifier still contributes work.
                    if greedy_mode:
                        corrected_ids = [_greedy_token_id_from_logits(current_next_logits)]
                    else:
                        corrected_ids = [
                            int(
                                _sample_token_tensor(
                                    current_next_logits,
                                    temperature=self.sample_temperature,
                                ).item()
                            )
                        ]
                    stop = self._is_stop_token(corrected_ids[-1])
                    if stop:
                        stop_reason = "eos"
        elif len(committed_ids) >= completion_limit:
            stop_reason = "completion_limit"
            stop = True

        rejected_draft_ids = (
            list(draft_ids[len(accepted_ids):])
            if draft_ids and len(accepted_ids) < len(draft_ids)
            else []
        )
        verify_compute_ms = (time.perf_counter() - verify_start) * 1000.0
        provisional_verify_payload = {
            "accepted_ids": accepted_ids,
            "corrected_ids": corrected_ids,
            "rejected_draft_ids": rejected_draft_ids,
            "stop": stop,
            "stop_reason": stop_reason,
        }
        downlink_bytes = self._estimate_collaboration_downlink_bytes(provisional_verify_payload)
        network["downlink_bytes"] = int(downlink_bytes)
        network["downlink_transfer_ms"] = self._bandwidth_delay_ms(
            downlink_bytes,
            self.network_downlink_bandwidth_mbps,
        )
        network["downlink_ms"] = float(network.get("downlink_base_ms", 0.0)) + float(
            network["downlink_transfer_ms"]
        )
        network["network_ms"] = float(network.get("uplink_ms", 0.0)) + float(network["downlink_ms"])
        network["network_half_ms"] = network["network_ms"] / 2.0
        if self.enable_network_sleep and network["downlink_ms"] > 0.0:
            time.sleep(network["downlink_ms"] / 1000.0)
        draft_fully_accepted = bool(
            draft_ids and len(accepted_ids) == len(draft_ids)
        )
        bonus_token_emitted = bool(
            corrected_ids and draft_fully_accepted
        )
        residual_correction_emitted = bool(
            corrected_ids and not draft_fully_accepted
        )
        return {
            "accepted_ids": accepted_ids,
            "corrected_ids": corrected_ids,
            "rejected_draft_ids": rejected_draft_ids,
            "decision_state": {
                "accepted_count": int(len(accepted_ids)),
                "drafted_count": int(len(draft_ids)),
                "base_cache_length": 0,
                "pending_materialization_count": 0,
                "materialized_commit_ids": list(committed_ids) + list(accepted_ids),
                "next_logits_after_accept": locals().get("next_logits_after_accept"),
            },
            "stop": stop,
            "stop_reason": stop_reason,
            "cloud_compute_ms": verify_compute_ms,
            "network_overhead_ms": float(network["network_ms"]),
            "round_stats": {
                "draft_count": int(len(draft_ids)),
                "accepted_count": int(len(accepted_ids)),
                "corrected_count": int(len(corrected_ids)),
                "rejected_draft_count": int(len(rejected_draft_ids)),
                "accepted_length": int(len(accepted_ids)),
                "stop_reason": stop_reason,
                "draft_fully_accepted": draft_fully_accepted,
                "bonus_token_emitted": bonus_token_emitted,
                "residual_correction_emitted": residual_correction_emitted,
                # Alias kept for downstream analysis scripts.
                "all_accepted": draft_fully_accepted,
                "had_correction": residual_correction_emitted,
            },
        }

    @torch.no_grad()
    def append_committed(
        self,
        session,
        committed_ids,
        accepted_ids=None,
        corrected_ids=None,
        decision_state=None,
    ):
        """
        Commit verifier round result back into verifier session/cache state.

        Core design choice:
        - crop verifier branch cache to trusted materialized prefix
        - do NOT eagerly forward correction tokens
        - keep correction tokens as logical-only suffix until next verify round

        Parameters
        ----------
        session : dict
            Verifier session object.
        committed_ids : list[int]
            Full logical committed completion after this round.
        accepted_ids : list[int] | None
            Draft tokens accepted in this round.
        corrected_ids : list[int] | None
            Residual correction / bonus tokens emitted in this round.
        decision_state : dict | None
            Expected structure is:
            `{"accepted_count": int, "base_cache_length": int,
              "pending_materialization_count": int,
              "materialized_commit_ids": list[int],
              "next_logits_after_accept": torch.Tensor | None}`.

        Returns
        -------
        dict
            Minimal commit-phase telemetry.
        """
        # Commit phase:
        # - crop the verifier branch cache back to the materialized committed prefix
        # - do NOT eagerly forward correction tokens
        #   correction tokens remain logical-only and are lazily materialized
        #   during the next verify round together with the next draft block
        session["committed_ids"] = list(committed_ids)
        accepted_ids = list(accepted_ids or [])
        corrected_ids = list(corrected_ids or [])
        request = session.get("request", {})
        self._ensure_loaded()

        crop_ms = 0.0

        materialized_commit_ids = list(committed_ids)
        if isinstance(decision_state, dict):
            materialized_commit_ids = list(
                decision_state.get("materialized_commit_ids", materialized_commit_ids)
            )
        rebuild_start = time.perf_counter()
        prefix = self._build_prefix(session["prompt_ids"].to(self.device), materialized_commit_ids)
        self._initialize_verify_session(
            self._get_request_id(request),
            prefix,
            materialized_commit_ids,
        )
        crop_ms = (time.perf_counter() - rebuild_start) * 1000.0

        return {"cloud_compute_ms": crop_ms}

    def _generate(self, prompt_ids, completion_limit, token_callback=None, request=None):
        """
        Run verifier-only autoregressive generation.

        Parameters
        ----------
        prompt_ids : torch.Tensor
            Shape `[1, prompt_len]`.
        completion_limit : int
            Maximum newly generated tokens.
        token_callback : Callable | None
            Optional callback invoked once per emitted token id.
        request : dict | None
            Request metadata used by stop post-processing.

        Returns
        -------
        tuple[list[int], float, float, dict]
            `(completion_ids, total_ms, ttft_ms, timestamp_bundle)`.
        """
        if self.generation_backend == "transformers":
            return self._generate_with_transformers(
                prompt_ids,
                completion_limit,
                token_callback=token_callback,
                request=request,
            )
        if self.generation_backend == "vllm":
            return self._generate_with_vllm(
                prompt_ids,
                completion_limit,
                token_callback=token_callback,
                request=request,
            )

        generated = []
        start = time.perf_counter()
        ttft_ms = None
        first_token_ns = None
        outputs = self.model(input_ids=prompt_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        stop_reason = "completion_limit"
        for _ in range(max(int(completion_limit), 0)):
            next_id = int(
                _sample_token_tensor(
                    next_logits,
                    temperature=self.sample_temperature,
                ).item()
            )
            generated.append(next_id)
            if token_callback:
                token_callback(next_id)
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000.0
            if first_token_ns is None:
                first_token_ns = now_ns()
            if self._is_stop_token(next_id):
                stop_reason = "eos"
                break
            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            outputs = self.model(
                input_ids=next_tensor,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
        total_ms = (time.perf_counter() - start) * 1000.0
        final_ids, final_stop_reason = apply_stop_to_sequence(
            self.tokenizer,
            generated,
            self.stop_mode,
            request=request,
        )
        return final_ids, total_ms, (ttft_ms or total_ms), {
            "first_token_ns": first_token_ns,
            "stop_reason": final_stop_reason or stop_reason,
        }

    def _generate_with_transformers(
        self,
        prompt_ids,
        completion_limit,
        token_callback=None,
        request=None,
    ):
        """
        Run cloud-only generation via `transformers.generate`.

        Parameters
        ----------
        prompt_ids : torch.Tensor
            Tokenized prompt ids on the verifier device.
        completion_limit : int
            Maximum number of new tokens to produce.
        token_callback : Callable | None, default=None
            Optional callback invoked once per generated token.
        request : dict | None, default=None
            Request metadata used by stop post-processing.

        Returns
        -------
        tuple[list[int], float, float, dict]
            Generation ids plus measured total / first-token latency metadata.
        """
        start_abs_ns = now_ns()
        start = time.perf_counter()
        generate_kwargs = {
            "input_ids": prompt_ids,
            "max_new_tokens": max(int(completion_limit), 0),
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.tokenizer.eos_token_id is not None:
            generate_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if is_greedy_temperature(self.sample_temperature):
            generate_kwargs["do_sample"] = False
        else:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = float(self.sample_temperature)

        outputs = self.model.generate(**generate_kwargs)
        total_ms = (time.perf_counter() - start) * 1000.0
        generated = outputs[0, prompt_ids.shape[1] :].tolist()
        for token_id in generated:
            if token_callback:
                token_callback(int(token_id))

        final_ids, final_stop_reason = apply_stop_to_sequence(
            self.tokenizer,
            generated,
            self.stop_mode,
            request=request,
        )
        first_token_ns = None
        if final_ids:
            first_token_ns = start_abs_ns + int(total_ms * 1_000_000)
        return final_ids, total_ms, total_ms, {
            "first_token_ns": first_token_ns,
            "stop_reason": final_stop_reason or "completion_limit",
        }

    def _generate_with_vllm(
        self,
        prompt_ids,
        completion_limit,
        token_callback=None,
        request=None,
    ):
        """
        Run cloud-only generation via the vLLM backend.

        Parameters
        ----------
        prompt_ids : torch.Tensor
            Tokenized prompt ids on the verifier device.
        completion_limit : int
            Maximum number of new tokens to produce.
        token_callback : Callable | None, default=None
            Optional callback invoked once per generated token.
        request : dict | None, default=None
            Request metadata used by stop post-processing.

        Returns
        -------
        tuple[list[int], float, float, dict]
            Generation ids plus measured total / first-token latency metadata.
        """
        if self.vllm_model is None:
            raise RuntimeError("vLLM backend requested before the verifier vLLM engine was loaded.")

        start_abs_ns = now_ns()
        start = time.perf_counter()
        sampling_params = build_vllm_sampling_params(
            completion_limit,
            self.sample_temperature,
            stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else None,
        )
        outputs = self.vllm_model.generate(
            prompts=[prompt_ids[0].tolist()],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        fallback_total_ms = (time.perf_counter() - start) * 1000.0
        request_output = outputs[0]
        output_item = request_output.outputs[0] if request_output.outputs else None
        generated = list(output_item.token_ids or []) if output_item is not None else []
        for token_id in generated:
            if token_callback:
                token_callback(int(token_id))

        total_ms, ttft_ms = extract_vllm_timing_ms(request_output, fallback_total_ms)
        final_ids, final_stop_reason = apply_stop_to_sequence(
            self.tokenizer,
            generated,
            self.stop_mode,
            request=request,
        )
        first_token_ns = None
        if final_ids:
            first_token_ns = start_abs_ns + int(ttft_ms * 1_000_000)
        finish_reason = str(getattr(output_item, "finish_reason", "") or "")
        return final_ids, total_ms, ttft_ms, {
            "first_token_ns": first_token_ns,
            "stop_reason": final_stop_reason or finish_reason or "completion_limit",
        }

    def _get_active_verify_session(self, request, prompt_ids, committed_ids):
        """
        Return the active verifier session for the current committed prefix.

        Parameters
        ----------
        request : dict
            Current normalized request.
        prompt_ids : torch.Tensor
            Prompt ids.
        committed_ids : list[int]
            Logical committed completion prefix.

        Returns
        -------
        dict
            Active verifier internal session from `self._verify_sessions`.

        Raises
        ------
        RuntimeError
            If the stored materialized prefix is not a prefix of `committed_ids`.
        """
        request_id = self._get_request_id(request)
        session = self._verify_sessions.get(request_id)

        if session is None:
            prefix = self._build_prefix(prompt_ids.to(self.device), committed_ids)
            return self._initialize_verify_session(request_id, prefix, committed_ids)

        materialized_ids = list(session.get("materialized_ids", []))
        expected_committed = list(committed_ids or [])
        if (
            len(materialized_ids) > len(expected_committed)
            or materialized_ids != expected_committed[: len(materialized_ids)]
        ):
            raise RuntimeError(
                "Verifier session is out of sync with committed tokens. "
                f"request_id={request_id}, "
                f"materialized_len={len(materialized_ids)}, "
                f"expected_committed_len={len(expected_committed)}."
            )

        return session

    def _initialize_verify_session(self, request_id, prefix, committed_ids):
        """
        Initialize verifier KV state from the committed prefix.

        Parameters
        ----------
        request_id : str
            Session key.
        prefix : torch.Tensor
            Full model prefix.
        committed_ids : list[int]
            Committed completion tokens already accepted logically.

        Returns
        -------
        dict
            Session object with keys `materialized_ids`, `past_key_values`,
            and `next_logits`.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=prefix,
                past_key_values=DynamicCache(),
                use_cache=True,
            )
        next_session = {
            "materialized_ids": list(committed_ids),
            "past_key_values": _ensure_dynamic_cache(outputs.past_key_values),
            "next_logits": outputs.logits[:, -1, :],
        }
        self._verify_sessions[request_id] = next_session
        return next_session

    def _next_token_id(self, input_ids):
        """
        Query verifier for one next-token sample.

        Parameters
        ----------
        input_ids : torch.Tensor
            Shape `[1, seq_len]`.

        Returns
        -------
        int
            Sampled next token id.
        """
        outputs = self.model(input_ids=input_ids, use_cache=True)
        return int(
            _sample_token_tensor(
                outputs.logits[:, -1, :],
                temperature=self.sample_temperature,
            ).item()
        )

    def _prepare_prompt_ids(self, request):
        """
        Tokenize request prompt into verifier input ids.

        Parameters
        ----------
        request : dict
            Normalized request. Must contain `query`.

        Returns
        -------
        torch.Tensor
            Shape `[1, prompt_len]` on verifier device.
        """

        tokenizer_kwargs = {
            "return_tensors": "pt",
        }
        if request.get("prompt_tokens") is not None:
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = int(request["prompt_tokens"])
        else:
            tokenizer_kwargs["truncation"] = False
        encoded = self.tokenizer(request["query"], **tokenizer_kwargs)
        prompt_token_count = int(encoded.input_ids.shape[1])
        context_limit = self._context_limit()
        if prompt_token_count >= context_limit:
            raise ValueError(
                f"Prompt length exceeds verify model context limit without truncation: "
                f"prompt_tokens={prompt_token_count}, context_limit={context_limit}."
            )
        return encoded.input_ids.to(self.device)

    def _build_prefix(self, prompt_ids, committed_ids):
        """
        Concatenate prompt ids and committed completion ids.

        Returns
        -------
        torch.Tensor
            Shape `[1, prompt_len + committed_len]`.
        """
        if not committed_ids:
            return prompt_ids
        committed_tensor = torch.tensor([committed_ids], dtype=torch.long, device=self.device)
        return torch.cat([prompt_ids, committed_tensor], dim=1)

    def _context_limit(self):
        """
        Resolve verifier model context limit.

        Returns
        -------
        int
            Maximum supported sequence length.
        """
        config = getattr(self.model, "config", None)
        candidates = [
            getattr(config, "max_position_embeddings", None),
            getattr(config, "max_sequence_length", None),
            getattr(config, "seq_length", None),
            getattr(self.tokenizer, "model_max_length", None),
        ]
        for value in candidates:
            if value is None:
                continue
            value = int(value)
            if 0 < value < 10**7:
                return value
        raise RuntimeError("Unable to determine verify model context limit.")

    def _resolve_completion_limit(self, prompt_token_count, requested_completion_tokens):
        """
        Resolve the natural generation budget from request or model context.

        If the request does not specify `completion_tokens`, we allow generation
        to continue until EOS or until the remaining context window is exhausted.
        """
        if requested_completion_tokens is not None:
            return max(1, int(requested_completion_tokens))
        context_limit = self._context_limit()
        remaining = context_limit - int(prompt_token_count) - 1
        if remaining <= 0:
            raise ValueError(
                f"Prompt is too long for verify model context limit: "
                f"prompt_tokens={prompt_token_count}, context_limit={context_limit}."
            )
        if self.runaway_guard_max_new_tokens is not None:
            return min(remaining, max(1, int(self.runaway_guard_max_new_tokens)))
        return remaining

    def _is_stop_token(self, token_id):
        """
        Check whether a token id should terminate generation.

        Returns
        -------
        bool
        """
        return is_stop_token(self.tokenizer, token_id)

    def _ensure_loaded(self, require_torch_model=True):
        """
        Lazily construct tokenizer and model resources.

        Parameters
        ----------
        require_torch_model : bool, default=True
            Whether the PyTorch `transformers` model must be loaded for the
            current execution path.

        Returns
        -------
        None
            Mutates `self.tokenizer` and `self.model`.
        """
        if self.tokenizer is not None and (
            not require_torch_model or self.model is not None or self.vllm_model is not None
        ):
            return
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not require_torch_model and self.generation_backend == "vllm":
            if self.vllm_model is None:
                LLM, _ = import_vllm()
                max_model_len = max(
                    2048,
                    int(self.default_prompt_tokens or 1024) + int(self.default_completion_tokens or 128) + 64,
                )
                self.vllm_model = LLM(
                    model=self.model_name,
                    tokenizer=self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    dtype="float16" if self.device == "cuda" else "float32",
                    tensor_parallel_size=1,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=0.90,
                    disable_log_stats=True,
                    enforce_eager=True,
                )
            return

        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.dtype,
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        self.model = self.model.to(self.device)
        if hasattr(self.model, "config"):

            self.model.config.use_cache = True
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            generation_config.use_cache = True
        self.model.eval()

    @torch.no_grad()
    def _estimate_collaboration_uplink_bytes(self, request, draft_ids, draft_logits):
        """
        Estimate edge->cloud payload size for collaboration mode.

        Parameters
        ----------
        request : dict
            Current request, mainly for `request_id`.
        draft_ids : list[int]
            Proposed draft block.
        draft_logits : list[torch.Tensor] | torch.Tensor
            Drafter q logits transported to verifier.

        Returns
        -------
        int
            Approximate uplink byte count.
        """
        request_id = ""
        if isinstance(request, dict):
            request_id = str(request.get("request_id", ""))
        return (
            self._estimate_tensor_bytes(request_id)
            + self._estimate_tensor_bytes(draft_ids)
            + self._estimate_tensor_bytes(draft_logits)
        )

    def _estimate_collaboration_downlink_bytes(self, verify_payload):
        """
        Estimate cloud->edge payload size for verified-round commit.

        Parameters
        ----------
        verify_payload : dict
            Verifier output payload containing accepted/corrected/rejected ids.

        Returns
        -------
        int
            Approximate downlink byte count.
        """
        verify_payload = verify_payload or {}
        return (
            self._estimate_tensor_bytes(verify_payload.get("accepted_ids", []))
            + self._estimate_tensor_bytes(verify_payload.get("corrected_ids", []))
            + self._estimate_tensor_bytes(verify_payload.get("rejected_draft_ids", []))
            + self._estimate_tensor_bytes(bool(verify_payload.get("stop", False)))
            + self._estimate_tensor_bytes(str(verify_payload.get("stop_reason", "")))
        )

    def _estimate_cloud_only_uplink_bytes(self, request, prompt_ids):
        """
        Estimate request upload size for cloud-only inference.

        Returns
        -------
        int
        """
        request = request or {}
        return (
            self._estimate_tensor_bytes(request.get("request_id", ""))
            + self._estimate_tensor_bytes(request.get("query", ""))
            + self._estimate_tensor_bytes(prompt_ids)
        )

    def _estimate_cloud_only_downlink_bytes(self, completion_ids):
        """
        Estimate completion download size for cloud-only inference.

        Returns
        -------
        int
        """
        return self._estimate_tensor_bytes(completion_ids)

    def _sample_network_delay(self, uplink_bytes=0, downlink_bytes=0):
        """
        Sample one benchmark-side network delay bundle.

        Parameters
        ----------
        uplink_bytes : int, default=0
            Estimated bytes from edge to cloud.
        downlink_bytes : int, default=0
            Estimated bytes from cloud to edge.

        Returns
        -------
        dict
            Network timing/size bundle with keys such as `uplink_ms`,
            `downlink_ms`, `network_ms`, `uplink_bytes`, `downlink_bytes`.
        """
        # Benchmark-only network simulator. Sedna itself does not provide network
        # emulation; the example layer turns configured RTT / jitter into per-round
        # uplink and downlink delays.
        jitter = (
            self.network_rng.uniform(-self.network_jitter_ms, self.network_jitter_ms)
            if self.network_jitter_ms > 0.0
            else 0.0
        )
        rtt_ms = max(self.network_rtt_ms + jitter, 0.0)
        uplink_base_ms = rtt_ms * self.network_uplink_ratio
        downlink_base_ms = rtt_ms - uplink_base_ms
        uplink_transfer_ms = self._bandwidth_delay_ms(
            uplink_bytes,
            self.network_uplink_bandwidth_mbps,
        )
        downlink_transfer_ms = self._bandwidth_delay_ms(
            downlink_bytes,
            self.network_downlink_bandwidth_mbps,
        )
        uplink_ms = uplink_base_ms + uplink_transfer_ms
        downlink_ms = downlink_base_ms + downlink_transfer_ms
        return {
            "network_rtt_ms": rtt_ms,
            "network_jitter_ms": abs(jitter),
            "network_rtt_base_ms": rtt_ms,
            "uplink_base_ms": uplink_base_ms,
            "downlink_base_ms": downlink_base_ms,
            "uplink_bytes": int(max(int(uplink_bytes or 0), 0)),
            "downlink_bytes": int(max(int(downlink_bytes or 0), 0)),
            "uplink_transfer_ms": uplink_transfer_ms,
            "downlink_transfer_ms": downlink_transfer_ms,
            "uplink_ms": uplink_ms,
            "downlink_ms": downlink_ms,
            "network_ms": uplink_ms + downlink_ms,
            "network_half_ms": (uplink_ms + downlink_ms) / 2.0,
        }

    @staticmethod
    def _estimate_tensor_bytes(value):
        """
        Estimate serialized size of a tensor-like payload.

        Parameters
        ----------
        value : Any
            Tensor, list, dict, scalar, or string-like payload.

        Returns
        -------
        int
            Approximate byte size used by benchmark network simulation.
        """
        if value is None:
            return 0
        if isinstance(value, torch.Tensor):
            return int(value.numel()) * int(value.element_size())
        if isinstance(value, (bytes, bytearray)):
            return len(value)
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        if isinstance(value, bool):
            return 1
        if isinstance(value, int):
            return 8
        if isinstance(value, float):
            return 8
        if isinstance(value, dict):
            return sum(
                SpeculativeVerifyModel._estimate_tensor_bytes(key)
                + SpeculativeVerifyModel._estimate_tensor_bytes(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return sum(SpeculativeVerifyModel._estimate_tensor_bytes(item) for item in value)
        return len(str(value).encode("utf-8"))

    @staticmethod
    def _bandwidth_delay_ms(payload_bytes, bandwidth_mbps):
        """
        Convert payload size and link bandwidth into delay.

        Returns
        -------
        float
            Transfer time in milliseconds.
        """
        payload_bytes = max(int(payload_bytes or 0), 0)
        bandwidth_mbps = max(float(bandwidth_mbps or 0.0), 0.0)
        if payload_bytes <= 0 or bandwidth_mbps <= 0.0:
            return 0.0
        return (payload_bytes * 8.0) / (bandwidth_mbps * 1_000_000.0) * 1000.0

    def _get_request_id(self, request):
        """
        Extract stable request id for verifier session indexing.

        Returns
        -------
        str
        """
        return str(request.get("request_id", "default"))
