"""
Draft-side runtime for the cloud-edge speculative decoding benchmark.
"""

import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPECDEC_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
for path in (MODULE_DIR, SPECDEC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from base_drafter import BaseSpeculativeDrafter
from common.config_utils import _to_bool, _to_int, _to_optional_int
from common.request_utils import build_single_path_response, compute_perf, normalize_request
from common.stop_utils import apply_stop_to_sequence, is_stop_token
from sedna.common.class_factory import ClassFactory, ClassType

os.environ["BACKEND_TYPE"] = "TORCH"

from result_builder import SpeculativeDecodingResultBuilder
from common.timeline_utils import now_ns, offset_ns

def _sample_token_tensor(logits, temperature=0.0):

    """
    Sample one token id from a logits tensor.

    Parameters
    ----------
    logits : torch.Tensor
        Shape is typically `[1, vocab]` for a single decoding position.
    temperature : float, default=0.0
        Sampling temperature shared with verifier-side strict speculative logic.

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
    Apply temperature scaling to logits.

    This helper exists so drafter-side q logits and sampling behavior use the
    exact same scaling rule.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model logits for a single decoding position.
    temperature : float, default=0.0
        Sampling temperature.

    Returns
    -------
    torch.Tensor
        Float tensor with the same semantic shape as `logits`.
    """
    scaled_temperature = max(float(temperature or 0.0), 1e-5)
    return logits.float() / scaled_temperature

@ClassFactory.register(ClassType.GENERAL, alias="SpeculativeDraftModel")
class SpeculativeDraftModel(BaseSpeculativeDrafter):
    """
    Draft runtime for speculative decoding.

    Responsibilities:
    - load and own the draft model
    - provide edge-only generation
    - maintain draft-side KV cache across speculative rounds
    - format collaboration results for the benchmark
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.kwargs = kwargs

        self.algorithm_name = "ar_spec"

        self.model_name = kwargs.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
        # Expose the configured model name through the environment for Sedna runtime integration.
        os.environ["model_path"] = self.model_name

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

        self.stop_mode = str(kwargs.get("stop_mode", "choice")).strip().lower().replace("-", "_")

        self.inference_mode = kwargs.get("inference_mode")
        if self.inference_mode not in {"collaboration", "cloud-only", "edge-only"}:
            raise ValueError(
                f"Unsupported inference_mode: {self.inference_mode}. "
                f"Expected one of collaboration/cloud-only/edge-only."
            )

        self.sample_output_log = kwargs.get("sample_output_log")

        self.tokenizer = None
        self.model = None

        self._draft_sessions = {}

        self.result_builder = SpeculativeDecodingResultBuilder(self, **kwargs)

    def load(self, *args, **kwargs):
        """
        Load tokenizer and draft model into memory.

        Parameters
        ----------
        *args, **kwargs
            Unused; kept to match the surrounding Ianvs/Sedna model lifecycle.

        Returns
        -------
        None
            Mutates `self.tokenizer` and `self.model`.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:

            self.tokenizer.pad_token = self.tokenizer.eos_token

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

    def build_request(self, data):
        """
        Normalize one Ianvs dataset sample into the drafter request schema.

        Parameters
        ----------
        data : dict
            Standardized dataset item, usually containing at least `query` and
            possibly `gold`, `request_id`, `task_name`, `completion_tokens`.

        Returns
        -------
        dict
            Request object with normalized fields such as:
            `request_id`, `query`, `gold`, `task_name`, `prompt_tokens`,
            `completion_tokens`.
        """
        return normalize_request(
            data,
            self.default_prompt_tokens,
            self.default_completion_tokens,
            _to_optional_int,
        )

    def resolve_inference_mode(self, request=None, data=None, inference_mode=None):
        """
        Resolve per-request run mode for the benchmark.

        Parameters
        ----------
        request : dict | None
            Already normalized request.
        data : dict | None
            Raw dataset item used only when `request` is not provided.
        inference_mode : str | None
            Optional explicit override.

        Returns
        -------
        str
            One of `collaboration`, `cloud-only`, `edge-only`.
        """
        if request is None and data is not None:
            request = self.build_request(data)
        value = inference_mode
        if value is None and isinstance(request, dict):
            value = request.get("inference_mode")
        if value is None:
            value = self.inference_mode
        if value not in {"collaboration", "cloud-only", "edge-only"}:
            raise ValueError(
                f"Unsupported inference_mode: {value}. "
                f"Expected one of collaboration/cloud-only/edge-only."
            )
        return value

    def prepare_prompt(self, request):
        """
        Tokenize the request prompt and expose prompt metadata.

        Parameters
        ----------
        request : dict
            Normalized request produced by `build_request`.

        Returns
        -------
        dict
            `{"prompt_ids": torch.Tensor, "prompt_token_count": int}`.
        """

        prompt_ids = self._prepare_prompt_ids(request)
        return {"prompt_ids": prompt_ids, "prompt_token_count": int(prompt_ids.shape[1])}

    def start_session(self, data=None, request=None, **kwargs):
        """
        Create a drafter session for one request.

        Parameters
        ----------
        data : dict | None
            Raw dataset item.
        request : dict | None
            Optional pre-normalized request.
        **kwargs
            Additional request/build parameters.

        Returns
        -------
        dict
            Session object containing request metadata, prompt ids, runtime
            state, and draft-side accumulation buffers.
        """
        if request is None:
            request = self.build_request(data)
        prompt_payload = self.prepare_prompt(request)
        request = dict(request)

        request["completion_tokens"] = self._resolve_completion_limit(
            prompt_payload["prompt_token_count"],
            request.get("completion_tokens"),
        )
        session = {
            "request_id": str(request.get("request_id", "default")),
            "request": request,
            "prompt_ids": prompt_payload["prompt_ids"],
            "prompt_token_count": int(prompt_payload["prompt_token_count"]),
            "completion_limit": int(request["completion_tokens"]),
            "committed_ids": [],
            "timestamps": {},
            "request_start_abs_ns": now_ns(),
        }
        session["state"] = self._init_session_state(session)
        return session

    def step(self, session, feedback=None, max_draft_tokens=None, **kwargs):
        """
        Run one collaboration step from the drafter perspective.

        Stage order:
        1. consume previous verifier feedback
        2. if not stopped, choose round window
        3. generate a new draft block

        Parameters
        ----------
        session : dict
            Drafter session.
        feedback : dict | None
            Previous-round verifier feedback.
        max_draft_tokens : int | None
            Optional external cap for current round.

        Returns
        -------
        dict
            Drafter step payload. When stopped, may contain only control data.
            Otherwise includes `draft_ids`, `draft_logits`, timing, and debug
            metadata for verifier consumption.
        """
        control = self._consume_feedback(session, feedback=feedback)
        if control.get("stop", False):
            return {
                "draft_ids": [],
                "draft_logits": [],
                "selected_window": 0,
                "control": control,
            }
        state = self._get_state(session)
        if max_draft_tokens is None:
            remaining = max(
                int(state.get("completion_tokens", session.get("completion_limit", 1)))
                - len(state.get("committed_ids", [])),
                0,
            )
            max_draft_tokens = min(int(self.draft_tokens_per_step), remaining)
        else:
            max_draft_tokens = max(int(max_draft_tokens or 0), 0)

        payload = self.draft_step(
            request=session["request"],
            prompt_ids=session["prompt_ids"],
            committed_ids=session["committed_ids"],
            window=max(int(max_draft_tokens or 0), 0),
        )
        payload["control"] = control
        return payload

    def close_session(self, session, request=None):
        """
        Finalize one request and build benchmark-facing result context.

        Parameters
        ----------
        session : dict
            Drafter session.
        request : dict | None
            Optional request override.

        Returns
        -------
        dict
            Final session summary used by the benchmark result builder.
        """

        request = request or session.get("request")
        self._consume_feedback(session, feedback=session.pop("_pending_feedback", None))
        result = session.get("_final_result")
        if result is None and session.get("_pending_final_result"):
            state = self._get_state(session)
            state["timestamps"].setdefault(
                "request_end_ns",
                offset_ns(session.get("request_start_abs_ns", now_ns()), now_ns()),
            )
            result = self.result_builder.build_collaboration_result(
                request=session.get("request", {}),
                state=state,
            )
            session["_final_result"] = result
            session["_pending_final_result"] = False
        self.finalize_request(request=request)
        return result

    def inference(
        self,
        data=None,
        request=None,
        prompt_ids=None,
        prompt_token_count=None,
        token_callback=None,
    ):
        """
        Run edge-only inference on the drafter model.

        Parameters
        ----------
        data : dict | None, default=None
            Raw dataset sample.
        request : dict | None, default=None
            Pre-normalized request.
        prompt_ids : torch.Tensor | None, default=None
            Optional tokenized prompt.
        prompt_token_count : int | None, default=None
            Optional prompt length.
        token_callback : Callable | None, default=None
            Optional token callback.

        Returns
        -------
        dict
            Benchmark-facing edge-only result.
        """
        if request is None:
            request = self.build_request(data)
        if prompt_ids is None or prompt_token_count is None:

            prompt_payload = self.prepare_prompt(request)
            prompt_ids = prompt_payload["prompt_ids"]
            prompt_token_count = prompt_payload["prompt_token_count"]
        return self._build_edge_only_response(
            prompt_ids,
            prompt_token_count,
            request,
            token_callback=token_callback,
        )

    def finalize_request(self, request=None, request_id=None):
        """
        Drop request-local drafter cache/session state.

        Parameters
        ----------
        request : dict | None
            Request carrying `request_id`.
        request_id : str | None
            Explicit session id override.

        Returns
        -------
        None
        """
        rid = request_id
        if rid is None and isinstance(request, dict):
            rid = request.get("request_id")
        if rid is not None:
            session = self._draft_sessions.pop(str(rid), None)
            if isinstance(session, dict):
                session.pop("pending_draft", None)

    def cleanup(self):
        """
        Release model/runtime resources owned by the drafter.

        Returns
        -------
        None
        """
        self.model = None
        self.tokenizer = None
        self._draft_sessions.clear()
    def draft_step(self, request, prompt_ids, committed_ids, window):
        """
        Execute one draft-side speculative round.

        Parameters
        ----------
        request : dict
            Normalized request.
        prompt_ids : torch.Tensor
            Shape `[1, prompt_len]`.
        committed_ids : list[int]
            Tokens already accepted/committed before this round.
        window : int
            Maximum number of draft tokens to propose this round.

        Returns
        -------
        dict
            Draft payload with keys:
            - `draft_ids`: list[int]
            - `draft_logits`: list[torch.Tensor] | torch.Tensor
            - `selected_window`: int
            - `edge_compute_ms`: float
        """
        prepare_start = time.perf_counter()
        session = self._get_active_draft_session(request, prompt_ids, committed_ids)
        prepare_session_ms = (time.perf_counter() - prepare_start) * 1000.0
        draft_start = time.perf_counter()
        draft_ids, draft_logits = self._draft_ids_from_session(session, window)
        draft_generate_ms = (time.perf_counter() - draft_start) * 1000.0
        edge_generation_ms = prepare_session_ms + draft_generate_ms
        return {
            "draft_ids": draft_ids,
            "draft_logits": draft_logits,
            "selected_window": int(window),
            "edge_compute_ms": edge_generation_ms,
        }

    def _consume_feedback(self, session, feedback=None):
        """
        Apply verifier feedback from the previous round to drafter state.

        This is the main boundary between verifier output and drafter-side state
        advancement in the slim paradigm protocol.

        Parameters
        ----------
        session : dict
            Drafter session.
        feedback : dict | None
            Usually shaped as:
            `{"draft_output": dict, "verify_output": dict}`.

        Returns
        -------
        dict
            Control summary including committed tokens, stop flags, and round
            trace information.
        """
        feedback = feedback or session.pop("_pending_feedback", None)
        if not feedback:
            return {
                "stop": False,
                "progress": True,
                "result": None,
            }

        state = self._get_state(session)
        draft_output = dict((feedback or {}).get("draft_output", {}) or {})
        verify_output = dict((feedback or {}).get("verify_output", {}) or {})

        draft_ids = list(draft_output.get("draft_ids", []))
        accepted_ids = list(verify_output.get("accepted_ids", []))
        corrected_ids = list(verify_output.get("corrected_ids", []))

        draft_commit_output = self.append_committed(
            session,
            draft_ids=draft_ids,
            accepted_ids=accepted_ids,
            corrected_ids=corrected_ids,
        )

        state["rounds"] = int(state.get("rounds", 0)) + 1
        state["drafted_ids"].extend(draft_ids)
        state["accepted_draft_tokens"] += len(accepted_ids)
        state["committed_ids"].extend(accepted_ids)
        state["committed_ids"].extend(corrected_ids)
        state["corrected_ids"].extend(corrected_ids)
        state["edge_compute_ms"] += float(draft_output.get("edge_compute_ms", 0.0) or 0.0)
        state["cloud_compute_ms"] += float(verify_output.get("cloud_compute_ms", 0.0) or 0.0)
        state["network_overhead_ms"] += float(verify_output.get("network_overhead_ms", 0.0) or 0.0)
        state["committed_token_trace"].extend(
            self._build_token_trace_entries(
                round_index=state["rounds"],
                draft_ids=draft_ids,
                accepted_ids=accepted_ids,
                corrected_ids=corrected_ids,
                verify_output=verify_output,
            )
        )
        if (accepted_ids or corrected_ids) and not state["timestamps"].get("first_token_ns"):
            state["timestamps"]["first_token_ns"] = offset_ns(
                session.get("request_start_abs_ns", now_ns()),
                now_ns(),
            )
        self._record_round(
            state,
            round_index=state["rounds"],
            draft_output=draft_output,
            verify_output=verify_output,
            draft_commit_output=draft_commit_output,
        )

        reached_completion_limit = (
            len(state["committed_ids"])
            >= int(state.get("completion_tokens", session.get("completion_limit", 1)))
        )
        stop = bool(verify_output.get("stop", False) or reached_completion_limit)
        progress = bool(accepted_ids or corrected_ids or draft_ids or stop)
        if stop:
            state["timestamps"]["request_end_ns"] = offset_ns(
                session.get("request_start_abs_ns", now_ns()),
                now_ns(),
            )
            session["_pending_final_result"] = True
        return {
            "stop": stop,
            "progress": progress,
            "result": None,
        }

    def append_committed(
        self,
        session,
        draft_ids,
        accepted_ids,
        corrected_ids,
    ):
        """
        Update the drafter session after verifier feedback is committed.

        Parameters
        ----------
        session : dict
            Drafter session.
        draft_ids : list[int]
            Tokens proposed in the current draft step.
        accepted_ids : list[int]
            Draft tokens accepted by the verifier.
        corrected_ids : list[int]
            Tokens emitted by the verifier after acceptance or rejection.

        Returns
        -------
        dict
            Commit metadata for the current update.
        """
        draft_ids = list(draft_ids or [])
        accepted_ids = list(accepted_ids or [])
        corrected_ids = list(corrected_ids or [])
        committed_ids = list(session.get("committed_ids", [])) + list(accepted_ids) + list(corrected_ids)
        session["committed_ids"] = committed_ids
        request = session.get("request", {})
        request_id = session.get("request_id", request.get("request_id"))
        if request_id is not None:
            internal_session = self._draft_sessions.get(str(request_id))
            if isinstance(internal_session, dict):
                pending_draft = dict(internal_session.get("pending_draft") or {})
                cache = internal_session.get("cache")
                accepted_count = min(len(accepted_ids), len(draft_ids))
                base_cache_length = int(
                    pending_draft.get(
                        "base_cache_length",
                        max(int(internal_session.get("cache_length", 0)) - len(draft_ids), 0),
                    )
                )
                target_cache_length = base_cache_length + accepted_count
                if cache is not None and hasattr(cache, "crop"):
                    cache.crop(target_cache_length)
                    internal_session["cache"] = cache
                    cached_committed = list(internal_session.get("committed_ids", []))
                    correction_ids = list(corrected_ids)
                    internal_session["committed_ids"] = cached_committed + accepted_ids
                    internal_session["cache_length"] = target_cache_length
                    if correction_ids:
                        correction_tensor = torch.tensor(
                            [correction_ids],
                            dtype=torch.long,
                            device=self.device,
                        )
                        with torch.no_grad():
                            correction_outputs = self.model(
                                input_ids=correction_tensor,
                                past_key_values=internal_session["cache"],
                                use_cache=True,
                            )
                        internal_session["cache"] = correction_outputs.past_key_values
                        internal_session["cache_length"] = target_cache_length + len(correction_ids)
                        internal_session["next_logits"] = correction_outputs.logits[:, -1, :]
                        internal_session["committed_ids"] = cached_committed + accepted_ids + correction_ids
                    elif accepted_count == len(draft_ids):
                        internal_session["next_logits"] = pending_draft.get("next_logits", internal_session.get("next_logits"))
                    internal_session["pending_draft"] = None
                else:
                    self._draft_sessions.pop(str(request_id), None)
        return {}

    def _record_round(
        self,
        state,
        round_index=0,
        draft_output=None,
        verify_output=None,
        draft_commit_output=None,
    ):
        """
        Append one round summary to the collaboration trace.

        Parameters
        ----------
        state : dict
            Mutable request state.
        round_index : int, default=0
            One-based collaboration round index.
        draft_output : dict | None
            Drafter output for the round.
        verify_output : dict | None
            Verifier output for the round.
        draft_commit_output : dict | None
            Commit-phase metadata.

        Returns
        -------
        None
        """
        del draft_commit_output
        trace = state.setdefault("trace", [])
        trace.append(
            {
                "round": int(round_index),
                "selected_window": int((draft_output or {}).get("selected_window", 0) or 0),
                "draft_count": int(len((draft_output or {}).get("draft_ids", []) or [])),
                "accepted_count": int(len((verify_output or {}).get("accepted_ids", []) or [])),
                "corrected_count": int(len((verify_output or {}).get("corrected_ids", []) or [])),
                "rejected_draft_count": int(len((verify_output or {}).get("rejected_draft_ids", []) or [])),
                "committed_count": int(
                    len((verify_output or {}).get("accepted_ids", []) or [])
                    + len((verify_output or {}).get("corrected_ids", []) or [])
                ),
                "accepted_length": int(len((verify_output or {}).get("accepted_ids", []) or [])),
                "stop_reason": str((verify_output or {}).get("stop_reason", "") or ""),
                "rejected_draft_tokens": self._build_token_items(
                    (verify_output or {}).get("rejected_draft_ids", []) or []
                ),
                "draft_fully_accepted": bool(
                    (draft_output or {}).get("draft_ids")
                    and len((verify_output or {}).get("accepted_ids", []) or [])
                    == len((draft_output or {}).get("draft_ids", []) or [])
                ),
                "bonus_token_emitted": bool(
                    (verify_output or {}).get("corrected_ids", [])
                    and (draft_output or {}).get("draft_ids")
                    and len((verify_output or {}).get("accepted_ids", []) or [])
                    == len((draft_output or {}).get("draft_ids", []) or [])
                ),
                "residual_correction_emitted": bool(
                    (verify_output or {}).get("corrected_ids", [])
                    and not (
                        (draft_output or {}).get("draft_ids")
                        and len((verify_output or {}).get("accepted_ids", []) or [])
                        == len((draft_output or {}).get("draft_ids", []) or [])
                    )
                ),
                "all_accepted": bool(
                    (draft_output or {}).get("draft_ids")
                    and len((verify_output or {}).get("accepted_ids", []) or [])
                    == len((draft_output or {}).get("draft_ids", []) or [])
                ),
                "had_correction": bool(
                    (verify_output or {}).get("corrected_ids", [])
                    and not (
                        (draft_output or {}).get("draft_ids")
                        and len((verify_output or {}).get("accepted_ids", []) or [])
                        == len((draft_output or {}).get("draft_ids", []) or [])
                    )
                ),
            }
        )

    def _build_edge_only_response(
        self,
        prompt_ids,
        prompt_token_count,
        request,
        token_callback=None,
    ):
        """
        Build the final result for edge-only inference.

        Parameters
        ----------
        prompt_ids : torch.Tensor
            Tokenized prompt.
        prompt_token_count : int
            Prompt length in tokens.
        request : dict
            Normalized request.
        token_callback : Callable | None, default=None
            Optional token callback.

        Returns
        -------
        dict
            Benchmark-facing edge-only result.
        """
        request_start_abs_ns = now_ns()

        edge_tokens, edge_total_ms, edge_ttft_ms, generation_timestamps = self._generate(
            prompt_ids,
            self._resolve_completion_limit(prompt_token_count, request.get("completion_tokens")),
            token_callback=token_callback,
            request=request,
        )
        request_end_abs_ns = now_ns()
        timestamps = {
            "request_end_ns": offset_ns(request_start_abs_ns, request_end_abs_ns),
            "first_token_ns": offset_ns(
                request_start_abs_ns,
                generation_timestamps["first_token_ns"] or request_end_abs_ns,
            ),
        }
        result = self.result_builder.build_single_path_result(
            request=request,
            prompt_token_count=prompt_token_count,
            completion_ids=edge_tokens,
            routed_to="edge-only",
            timestamps=timestamps,
            extra_simulation={
                "stop_reason": generation_timestamps.get("stop_reason", ""),
            },
        )
        return result

    def _generate(self, prompt_ids, completion_limit, token_callback=None, request=None):
        """
        Run edge-only autoregressive generation on the drafter model.

        Parameters
        ----------
        prompt_ids : torch.Tensor
            Shape `[1, prompt_len]`.
        completion_limit : int
            Max number of newly generated tokens.
        token_callback : Callable | None
            Optional callback receiving emitted token ids during generation.
        request : dict | None
            Request metadata for debugging / stop handling.

        Returns
        -------
        tuple[list[int], float, float, dict]
            `(completion_ids, compute_ms, ttft_ms, generation_timestamps)`.
            `generation_timestamps` contains the first-token timestamp and stop reason.
        """

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

    def _emit_tokens(self, token_ids, token_callback, timeline):
        """
        Emit tokens and record the first visible-token timestamp.

        Parameters
        ----------
        token_ids : Sequence[int]
            Tokens to emit.
        token_callback : Callable | None
            Optional callback invoked per token id.
        timeline : list | None
            Optional mutable timeline list to append per-token events into.

        Returns
        -------
        None
        """
        if not token_ids:
            return
        for token_id in token_ids:
            if timeline.get("first_token_ns") is None:
                timeline["first_token_ns"] = offset_ns(
                    timeline["request_start_abs_ns"],
                    now_ns(),
                )
            if token_callback:
                token_callback(token_id)

    def _init_session_state(self, draft_session):
        """
        Initialize mutable per-request runtime state for the drafter.

        Parameters
        ----------
        draft_session : dict
            Session returned by `start_session`.

        Returns
        -------
        dict
            Runtime state dictionary stored under `session["runtime_state"]`.
        """
        request = dict((draft_session or {}).get("request", {}))
        return {
            "request_id": str(request.get("request_id", "unknown")),
            "request": request,
            "completion_tokens": max(int(request.get("completion_tokens", 1)), 1),
            "prompt_token_count": int((draft_session or {}).get("prompt_token_count", 0)),
            "committed_ids": [],
            "drafted_ids": [],
            "corrected_ids": [],
            "committed_token_trace": [],
            "accepted_draft_tokens": 0,
            "rounds": 0,
            "trace": [],
            "timestamps": dict((draft_session or {}).get("timestamps", {}) or {}),
            "edge_compute_ms": 0.0,
            "cloud_compute_ms": 0.0,
            "network_overhead_ms": 0.0,
            "meta": {},
        }

    def _get_state(self, session):
        """
        Retrieve the drafter runtime-state dictionary from a session.

        Parameters
        ----------
        session : dict
            Drafter session.

        Returns
        -------
        dict
            Session runtime state.
        """
        state = session.get("state")
        if state is None:
            state = self._init_session_state(session)
            session["state"] = state
        return state

    def _get_active_draft_session(self, request, prompt_ids, committed_ids):
        """
        Return the active drafter session for the current committed prefix.

        Parameters
        ----------
        request : dict
            Current normalized request.
        prompt_ids : torch.Tensor
            Prompt token ids.
        committed_ids : list[int]
            Already committed completion tokens.

        Returns
        -------
        dict
            Active draft session object stored in `self._draft_sessions`.

        Raises
        ------
        RuntimeError
            If the stored session prefix no longer matches `committed_ids`.
        """
        request_id = self._get_request_id(request)
        session = self._draft_sessions.get(request_id)

        if session is None:
            prefix = self._build_prefix(prompt_ids, committed_ids)
            return self._initialize_draft_session(request_id, prefix, committed_ids)

        cached_committed = list(session.get("committed_ids", []))
        expected_committed = list(committed_ids or [])
        if cached_committed != expected_committed:
            raise RuntimeError(
                "Draft session is out of sync with committed tokens. "
                f"request_id={request_id}, "
                f"cached_committed_len={len(cached_committed)}, "
                f"expected_committed_len={len(expected_committed)}."
            )
        return session

    @torch.no_grad()
    def _initialize_draft_session(self, request_id, prefix, committed_ids):
        """
        Initialize draft-side KV cache from the committed prefix.

        Parameters
        ----------
        request_id : str
            Session key.
        prefix : torch.Tensor
            Shape `[1, prefix_len]`, prompt + committed token ids.
        committed_ids : list[int]
            Tokens already committed by verifier.

        Returns
        -------
        dict
            Session object with keys:
            `committed_ids`, `cache`, `cache_length`, `next_logits`,
            `pending_draft`.
        """
        cache = DynamicCache()
        with torch.no_grad():
            outputs = self.model(
                input_ids=prefix,
                past_key_values=cache,
                use_cache=True,
            )

        next_session = {
            "committed_ids": list(committed_ids),
            "cache": outputs.past_key_values,
            "cache_length": int(prefix.shape[1]),
            "next_logits": outputs.logits[:, -1, :],
            "pending_draft": None,
        }
        self._draft_sessions[request_id] = next_session
        return next_session

    def _draft_ids_from_session(self, session, window):
        """
        Generate up to `window` speculative tokens from current draft session.

        Parameters
        ----------
        session : dict
            Drafter session object created by `_sync_draft_session`.
        window : int
            Maximum draft block size.

        Returns
        -------
        tuple[list[int], list[torch.Tensor]]
            Draft token ids and per-step q logits aligned by position.
        """

        if window <= 0:
            session["pending_draft"] = None
            return [], []
        draft_tokens = []
        draft_logits = []
        greedy_mode = float(self.sample_temperature or 0.0) < 1e-5
        local_cache = session["cache"]

        next_logits = session["next_logits"]
        base_cache_length = int(session.get("cache_length", 0))
        for _ in range(window):

            if not greedy_mode:
                draft_logits.append(
                    _scaled_logits(next_logits, temperature=self.sample_temperature).detach()
                )
            token_tensor = _sample_token_tensor(
                next_logits,
                temperature=self.sample_temperature,
            )
            draft_tokens.append(token_tensor)
            outputs = self.model(
                input_ids=token_tensor,
                past_key_values=local_cache,
                use_cache=True,
            )

            local_cache = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            if self._is_stop_token(int(token_tensor.item())):
                break
        draft_tensor = torch.cat(draft_tokens, dim=1)
        draft_ids = draft_tensor[0].tolist()
        session["cache"] = local_cache
        session["cache_length"] = base_cache_length + len(draft_ids)
        session["next_logits"] = next_logits
        batched_draft_logits = (
            torch.cat(draft_logits, dim=0) if draft_logits else None
        )
        session["pending_draft"] = {
            "draft_ids": list(draft_ids),
            "base_cache_length": base_cache_length,
            "next_logits": next_logits,
            "draft_logits": batched_draft_logits,
        }
        return draft_ids, batched_draft_logits

    @torch.no_grad()
    def _prepare_prompt_ids(self, request):
        """
        Tokenize prompt text into device-local input ids.

        Parameters
        ----------
        request : dict
            Normalized request. Must include `query`; may include
            `prompt_tokens` as a truncation budget.

        Returns
        -------
        torch.Tensor
            Shape `[1, prompt_len]` on `self.device`.

        Raises
        ------
        ValueError
            If prompt length already exceeds model context capacity.
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
                f"Prompt length exceeds draft model context limit without truncation: "
                f"prompt_tokens={prompt_token_count}, context_limit={context_limit}."
            )
        return encoded.input_ids.to(self.device)

    def _build_prefix(self, prompt_ids, committed_ids):
        """
        Build the full model prefix `prompt + committed completion`.

        Parameters
        ----------
        prompt_ids : torch.Tensor
            Prompt ids with shape `[1, prompt_len]`.
        committed_ids : list[int]
            Already committed completion tokens.

        Returns
        -------
        torch.Tensor
            Shape `[1, prompt_len + len(committed_ids)]`.
        """

        if not committed_ids:
            return prompt_ids
        committed = torch.tensor([committed_ids], dtype=torch.long, device=self.device)
        return torch.cat([prompt_ids, committed], dim=1)

    def decode_tokens(self, token_ids):
        """
        Decode token ids to display text.

        Parameters
        ----------
        token_ids : Sequence[int]
            Token ids to decode.

        Returns
        -------
        str
            Decoded text with special tokens removed.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _build_token_items(self, token_ids):
        """
        Convert token ids into `(id, text)` items for debug / result payloads.

        Parameters
        ----------
        token_ids : Sequence[int]
            Token ids to decode one-by-one.

        Returns
        -------
        list[dict]
            Each item has keys `token_id` and `text`.
        """
        items = []
        for token_id in list(token_ids or []):
            token_id = int(token_id)
            items.append(
                {
                    "token_id": token_id,
                    "token_text": self.decode_tokens([token_id]),
                }
            )
        return items

    def _context_limit(self):
        """
        Resolve the effective context window from model/tokenizer metadata.

        Returns
        -------
        int
            Maximum supported sequence length.

        Raises
        ------
        RuntimeError
            If no valid context limit can be derived.
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
        raise RuntimeError("Unable to determine draft model context limit.")

    def _resolve_completion_limit(self, prompt_token_count, requested_completion_tokens):
        """
        Resolve how many new tokens may be generated.

        We no longer use benchmark-imposed fixed lengths by default. When the
        request does not specify a completion budget, we allow generation to
        continue naturally until EOS or until the model context window is full.
        """
        if requested_completion_tokens is not None:
            return max(1, int(requested_completion_tokens))
        context_limit = self._context_limit()
        remaining = context_limit - int(prompt_token_count) - 1
        if remaining <= 0:
            raise ValueError(
                f"Prompt is too long for draft model context limit: "
                f"prompt_tokens={prompt_token_count}, context_limit={context_limit}."
            )
        if self.runaway_guard_max_new_tokens is not None:
            return min(remaining, max(1, int(self.runaway_guard_max_new_tokens)))
        return remaining

    def _is_stop_token(self, token_id):
        """
        Check whether a token id is a model-native stop token.

        Parameters
        ----------
        token_id : int
            Generated token id.

        Returns
        -------
        bool
            Whether generation should stop immediately after this token.
        """
        return is_stop_token(self.tokenizer, token_id)

    def _get_request_id(self, request):
        """
        Extract the stable request id used as drafter-session key.

        Parameters
        ----------
        request : dict
            Normalized request.

        Returns
        -------
        str
            Stable request/session identifier.
        """
        return str(request.get("request_id", "default"))

    @staticmethod
    def _label_corrected_token_source(draft_ids, accepted_ids, verify_output):
        """
        Classify the provenance of verifier-emitted tokens for tracing.

        Parameters
        ----------
        draft_ids : Sequence[int]
            Token ids proposed by the drafter in the current round.
        accepted_ids : Sequence[int]
            Prefix of `draft_ids` accepted by the verifier.
        verify_output : dict | None
            Verifier round payload. Expected keys may include
            `round_stats.bonus_token_emitted` and
            `round_stats.residual_correction_emitted`.

        Returns
        -------
        str
            `"verifier_bonus"` or `"verifier_correction"`.
        """
        round_stats = (verify_output or {}).get("round_stats", {}) or {}
        if "bonus_token_emitted" in round_stats:
            return "verifier_bonus" if round_stats.get("bonus_token_emitted") else "verifier_correction"
        if "residual_correction_emitted" in round_stats:
            return "verifier_correction" if round_stats.get("residual_correction_emitted") else "verifier_bonus"
        draft_ids = list(draft_ids or [])
        accepted_ids = list(accepted_ids or [])
        return "verifier_bonus" if draft_ids and len(accepted_ids) == len(draft_ids) else "verifier_correction"

    @classmethod
    def _build_token_trace_entries(cls, *, round_index, draft_ids, accepted_ids, corrected_ids, verify_output):
        """
        Build token-level provenance entries for one speculative round.

        Parameters
        ----------
        round_index : int
            Zero-based speculative round index.
        draft_ids : Sequence[int]
            All tokens proposed by the drafter.
        accepted_ids : Sequence[int]
            Tokens accepted from the draft block.
        corrected_ids : Sequence[int]
            Tokens emitted by verifier after acceptance / rejection handling.
        verify_output : dict | None
            Full verifier payload for this round.

        Returns
        -------
        list[dict]
            Each item has shape:
            `{"round": int, "token_id": int, "source": str}`.
        """
        entries = []
        for token_id in list(accepted_ids or []):
            entries.append(
                {
                    "round": int(round_index),
                    "token_id": int(token_id),
                    "source": "draft_accepted",
                }
            )
        corrected_source = cls._label_corrected_token_source(
            draft_ids=draft_ids,
            accepted_ids=accepted_ids,
            verify_output=verify_output,
        )
        for token_id in list(corrected_ids or []):
            entries.append(
                {
                    "round": int(round_index),
                    "token_id": int(token_id),
                    "source": corrected_source,
                }
            )
        return entries
