"""Make one bounded direct loopback request; see bench/AGENTS.md."""

import http.client
import json
import socket
import threading
import time

from .files import decode_json
from .manifest import endpoint


CAPABILITIES = {"generation": True, "streaming": False, "likelihood": False,
                "ttft": False, "decode_timing": False, "memory": False}


def _close_socket(sock, expired):
    expired.set()
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


def _parse_completion(value, expected_model):
    if not isinstance(value, dict) or not isinstance(value.get("choices"), list):
        raise ValueError("Response has no completion choices")
    if len(value["choices"]) != 1 or not isinstance(value["choices"][0], dict):
        raise ValueError("Response must contain exactly one completion")
    choice = value["choices"][0]
    output, reason = choice.get("text"), choice.get("finish_reason")
    if not isinstance(output, str) or not isinstance(reason, str) or not reason:
        raise ValueError("Response must contain text and a terminal finish_reason")
    model = value.get("model")
    if model is not None and model != expected_model:
        raise ValueError("Server returned an unexpected model identifier")
    usage = value.get("usage")
    tokens = {"prompt_tokens": None, "completion_tokens": None}
    if usage is not None:
        if not isinstance(usage, dict):
            raise ValueError("Response usage must be an object")
        for key in tokens:
            reported = usage.get(key)
            if reported is not None and (type(reported) is not int or reported < 0):
                raise ValueError("Invalid reported token count")
            tokens[key] = reported
    return {"output": output, "finish_reason": reason, "truncated": reason == "length",
            "server_model": model, "reported_native_tokens": tokens,
            "token_count_source": "server_reported_unverified"}


def complete(manifest, prompt, timeout):
    """Return terminal completion text and full HTTP request latency."""
    backend, generation = manifest["backend"], manifest["generation"]
    host, port = endpoint(backend["endpoint"])
    payload = {"model": backend["model_alias"], "prompt": prompt, "stream": False,
               "n": 1, **generation}
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    started, watchdog = time.perf_counter(), None
    expired = threading.Event()
    try:
        connection.connect()
        remaining = timeout - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError("Request deadline exceeded while connecting")
        watchdog = threading.Timer(remaining, _close_socket, (connection.sock, expired))
        watchdog.daemon = True
        watchdog.start()
        connection.request("POST", "/v1/completions", body=body,
                           headers={"Content-Type": "application/json", "Connection": "close"})
        response = connection.getresponse()
        with response:
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}; redirects and non-200 replies are rejected")
            maximum = manifest["budget"]["max_response_bytes"]
            raw = response.read(maximum + 1)
            if expired.is_set():
                raise TimeoutError("Request deadline exceeded")
            if len(raw) > maximum:
                raise ValueError("Response body exceeds byte budget")
        duration = time.perf_counter() - started
        result = _parse_completion(decode_json(raw), backend["model_alias"])
        result.update(end_to_end_seconds=duration, output_utf8_bytes=len(result["output"].encode("utf-8")),
                      ttft_seconds=None, decode_tokens_per_second=None, peak_rss_bytes=None,
                      peak_vram_bytes=None, nll=None, bits_per_byte=None)
        return result
    finally:
        if watchdog is not None:
            watchdog.cancel()
            watchdog.join()
        connection.close()
