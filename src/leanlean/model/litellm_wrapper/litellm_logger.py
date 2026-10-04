# custom_callbacks.py
from leanlean.utils import json_utils as json
import hashlib
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
import traceback

# Import the CustomLogger from litellm
from litellm.integrations.custom_logger import CustomLogger
import pickle as pkl

DEBUG = False

# Log path configuration
LOG_PATH = "logs/litellm_server/traces/"   
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def _json_default(o):
    """
    Helper to handle non-serializable objects (datetime, Pydantic models, etc.)
    """
    if isinstance(o, timedelta):
        return o.total_seconds()
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, bytes):
        return o.decode("utf-8", "replace")
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return list(o)
    
    # Handle Pydantic v2
    if hasattr(o, "model_dump"):
        return o.model_dump()
    # Handle Pydantic v1
    if hasattr(o, "dict"):
        return o.dict()
    
    # Handle NamedTuples
    if hasattr(o, "_asdict"):
        return o._asdict()

    # Handle Dataclasses
    try:
        import dataclasses
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
    except Exception:
        pass
    
    if hasattr(o, "__dict__"):
        return o.__dict__
    
    # Fallback
    return repr(o)

class FileLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.log_path = LOG_PATH

    def _trace_file_path(self):
        trace_id = os.getenv("LEANLEAN_LITELLM_TRACE_ID", "").strip()
        if trace_id and trace_id.isascii() and trace_id.isdigit():
            return f"{self.log_path}/proxy_{trace_id}.json"
        return f"{self.log_path}/proxy_unscoped.json"

    def _cost_trace_file_path(self):
        trace_id = os.getenv("LEANLEAN_LITELLM_TRACE_ID", "").strip()
        if trace_id and trace_id.isascii() and trace_id.isdigit():
            return f"{self.log_path}/proxy_{trace_id}.cost.jsonl"
        return f"{self.log_path}/proxy_unscoped.cost.jsonl"

    def _extract_request_data(self, kwargs):
        """
        Helper to robustly extract the input data regardless of model type.
        Prioritizes: messages (Chat) -> input (Embedding) -> prompt (Text)
        """
        return (
            kwargs.get("messages") 
            or kwargs.get("input") 
            or kwargs.get("prompt") 
            or []
        )

    @staticmethod
    def _sanitize_request_value(value):
        """Redact credentials while preserving behavior-defining request data."""

        secret_markers = (
            "api_key",
            "apikey",
            "authorization",
            "access_token",
            "refresh_token",
            "id_token",
        )
        if isinstance(value, dict):
            return {
                str(key): (
                    "<redacted>"
                    if any(marker in str(key).lower() for marker in secret_markers)
                    else FileLogger._sanitize_request_value(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [FileLogger._sanitize_request_value(item) for item in value]
        if isinstance(value, tuple):
            return [FileLogger._sanitize_request_value(item) for item in value]
        return value

    def _extract_request_envelope(self, kwargs, request_data):
        """Capture all non-secret request fields omitted by the legacy field."""

        litellm_params = kwargs.get("litellm_params", {}) or {}
        proxy_request = litellm_params.get("proxy_server_request", {}) or {}
        body = proxy_request.get("body") or proxy_request.get("data")
        source = "proxy_server_request.body"
        complete = isinstance(body, dict)
        if not complete:
            source = "callback_kwargs"
            optional = kwargs.get("optional_params", {}) or {}
            body = {
                key: kwargs[key]
                for key in (
                    "model",
                    "messages",
                    "input",
                    "prompt",
                    "system",
                    "tools",
                    "tool_choice",
                    "parallel_tool_calls",
                    "temperature",
                    "top_p",
                    "max_tokens",
                    "max_completion_tokens",
                    "reasoning_effort",
                    "stream",
                    "metadata",
                )
                if key in kwargs
            }
            if isinstance(optional, dict):
                for key, value in optional.items():
                    body.setdefault(key, value)
        sanitized = self._sanitize_request_value(body)
        input_field = next(
            (
                key
                for key in ("messages", "input", "prompt")
                if isinstance(sanitized, dict) and key in sanitized
            ),
            None,
        )
        extras = (
            {
                key: value
                for key, value in sanitized.items()
                if key not in {"messages", "input", "prompt"}
            }
            if isinstance(sanitized, dict)
            else {}
        )
        canonical = json.dumps(
            sanitized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=_json_default,
        ).encode("utf-8", errors="surrogateescape")
        return {
            "schema": "leanlean-request-envelope-v1",
            "source": source,
            "complete": complete,
            "input_field": input_field,
            "extras": extras,
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "input_matches_request": (
                input_field is not None
                and isinstance(sanitized, dict)
                and sanitized[input_field] == request_data
            ),
        }

    @staticmethod
    def _response_field(response_obj, name, default=None):
        if isinstance(response_obj, dict):
            return response_obj.get(name, default)
        return getattr(response_obj, name, default)

    def _accounting_response(self, model_name, response_obj):
        """Retain exactly the response fields consumed by cost accounting."""

        response_model = self._response_field(response_obj, "model") or model_name
        usage = self._response_field(response_obj, "usage")
        response = {
            "model": response_model,
            "usage": usage if usage is not None else {},
        }
        response_id = self._response_field(response_obj, "id")
        if response_id:
            response["id"] = response_id
        return response

    def _extract_agent_context(self, kwargs):
        """Retain Claude's non-secret per-session/subagent routing IDs only."""
        litellm_params = kwargs.get("litellm_params", {})
        proxy_request = litellm_params.get("proxy_server_request", {}) or {}
        raw_headers = proxy_request.get("headers", {}) or {}
        headers = {str(key).lower(): value for key, value in raw_headers.items()}
        names = {
            "accounting_scope": "x-leanlean-accounting-scope",
            "session_id": "x-claude-code-session-id",
            "agent_id": "x-claude-code-agent-id",
            "parent_agent_id": "x-claude-code-parent-agent-id",
        }
        return {
            field: headers[header]
            for field, header in names.items()
            if headers.get(header)
        }


    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        
        if DEBUG:
            pkl_path = f"{self.log_path}/pkl_kwargs.pkl"
            response_obj_path = f"{self.log_path}/pkl_response_obj.pkl"
            with open(pkl_path, "ab") as pf:
                pkl.dump(kwargs, pf)
            with open(response_obj_path, "ab") as rf:
                pkl.dump(response_obj, rf)
        
        try:
            model_name = kwargs.get("model", "unknown_model")

            request_data = self._extract_request_data(kwargs)
            request_envelope = self._extract_request_envelope(
                kwargs, request_data
            )
            event_ts = time.time()
            call_id = kwargs.get("litellm_call_id", "")

            rec = {
                "ts": event_ts,
                "event": "success",
                "model": model_name,
                "request": request_data,
                "request_envelope": request_envelope,
                "response": response_obj,
                "litellm_call_id": call_id,
                "agent_context": self._extract_agent_context(kwargs),
            }
            provider_usage = kwargs.get("leanlean_provider_usage")
            if hasattr(provider_usage, "model_dump"):
                provider_usage = provider_usage.model_dump()
            accounting_response = self._accounting_response(model_name, response_obj)
            if isinstance(provider_usage, dict):
                # The pre-normalization provider payload is authoritative.
                # LiteLLM can overwrite streaming usage with locally counted
                # totals that omit cache and reasoning details.
                accounting_response["usage"] = provider_usage
            accounting_rec = {
                "ts": event_ts,
                "event": "success",
                "model": model_name,
                "response": accounting_response,
                "litellm_call_id": call_id,
                "agent_context": self._extract_agent_context(kwargs),
            }
                
            normalized_usage = self._response_field(response_obj, "usage")
            if hasattr(normalized_usage, "model_dump"):
                normalized_usage = normalized_usage.model_dump()
            normalized_usage = normalized_usage if isinstance(normalized_usage, dict) else {}
            authoritative_usage = (
                provider_usage if isinstance(provider_usage, dict) else normalized_usage
            )
            details = authoritative_usage.get("prompt_tokens_details") or {}
            provenance = {
                "source": "provider_stream" if provider_usage is not None else "litellm_normalized_unverified",
                "cache_usage_reported": details.get("cached_tokens") is not None,
                "billing_basis": "benchmark_rate_estimate_not_invoice",
            }
            for journal_record in (rec, accounting_rec):
                journal_record["usage_provenance"] = provenance
                if kwargs.get("leanlean_raw_stream_id"):
                    journal_record["raw_provider_stream_id"] = kwargs["leanlean_raw_stream_id"]
                if provider_usage is not None:
                    journal_record["provider_usage"] = provider_usage

            file_path = self._trace_file_path()
            cost_file_path = self._cost_trace_file_path()


            try:
                json_dump = json.dumps(rec, default=_json_default)
            except Exception as e:
                simple_response = {
                    "choices": [{"message": {"role": choice.message.role, "content": choice.message.content}} for choice in response_obj.choices]
                }
                                    
                rec = {
                    "ts": event_ts,
                    "event": "success",
                    "model": model_name,
                    "request": request_data,
                    "request_envelope": request_envelope,
                    "response": simple_response,
                    "litellm_call_id": call_id,
                    "agent_context": self._extract_agent_context(kwargs),
                }
            
                json_dump  = json.dumps(rec, default=_json_default)
            
            # Use the robust serializer
            with open(file_path, "a", buffering=1) as f:
                f.write(json_dump + "\n")
            with open(cost_file_path, "a", buffering=1) as f:
                f.write(json.dumps(accounting_rec, default=_json_default) + "\n")
                
            
        except Exception as e:
            # Dumb the error to a file
            if DEBUG:
                error_path = f"{self.log_path}/logger_errors.txt"
                with open(error_path, "a") as ef:
                    ef.write(f"Error logging success event: {str(e)}\n")
                    ef.write(traceback.format_exc() + "\n")

            # Print error to stderr so you know if the logger itself is failing
            print(f"FileLogger Error (Success Event): {e}")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):

        if DEBUG:
            pkl_path = f"{self.log_path}/pkl_kwargs_fail.pkl"
            with open(pkl_path, "ab") as pf:
                pkl.dump(kwargs, pf)

        try:
            model_name = kwargs.get("model", "unknown_model")

            request_data = self._extract_request_data(kwargs)
            request_envelope = self._extract_request_envelope(
                kwargs, request_data
            )
            
            rec = {
                "ts": time.time(),
                "event": "failure",
                "model": model_name,
                "request": request_data,
                "request_envelope": request_envelope,
                "litellm_call_id": kwargs.get("litellm_call_id", ""),
                "response": response_obj,
                "exception": str(kwargs.get("exception", "")), 
                "agent_context": self._extract_agent_context(kwargs),
            }

            file_path = self._trace_file_path()
            
            # Use the robust serializer
            with open(file_path, "a", buffering=1) as f:
                f.write(json.dumps(rec, default=_json_default) + "\n")
                
        except Exception as e:
            print(f"FileLogger Error (Failure Event): {e}")

# Instance to be used in config
file_logger = FileLogger()
