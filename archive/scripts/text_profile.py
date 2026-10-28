"""Pin the T2 development profile; see docs/T2-CREDIT-DIAGNOSTIC.md."""

import hashlib


T2_TRAIN_SHA256 = "b58add621357506a1eded9867080a76f7015789067ed089fc47818e424230f8a"
T2_DEVELOPMENT_SHA256 = "e3ae44c3a53ae682cbf42ed18a40659b29d1a3b307afb25ecb1bfd5882117032"
T2_PREFIX_SHA256 = "40698f4235a3ab2316155ac9b8762b103d0a49bca5b479485b3b75119260e2c2"
T2_PREFIX_FNV = "1ec64f408ae9ca39"


def validate_profile(profile: str, diagnostics: str, mode: str) -> None:
    if profile not in ("fixture", "t2-credit") or diagnostics not in ("off", "on"):
        raise ValueError("unsupported text profile or credit diagnostics")
    if diagnostics == "on" and profile != "t2-credit":
        raise ValueError("credit diagnostics require explicit t2-credit profile")
    if profile == "t2-credit" and mode != "pulse":
        raise ValueError("T2 requires pulse mode")


def evaluation_path(profile: str) -> str:
    return "datasets/tiny/development-t2.txt" if profile == "t2-credit" else "datasets/tiny/valid.txt"


def input_identity(paths: dict, hashes: dict, profile: str) -> dict:
    if profile == "t2-credit" and (hashes["train"] != T2_TRAIN_SHA256 or hashes["valid"] != T2_DEVELOPMENT_SHA256):
        raise ValueError("T2 requires frozen train/development identities")
    prefix = paths["valid"].read_bytes()[:72]
    identity = {"profile": profile, "role": "development" if profile == "t2-credit" else "held-out-fixture",
                "relative_path": evaluation_path(profile), "offset": 0, "byte_count": len(prefix),
                "prefix_sha256": hashlib.sha256(prefix).hexdigest()}
    if profile == "t2-credit" and (len(prefix) != 72 or identity["prefix_sha256"] != T2_PREFIX_SHA256):
        raise ValueError("T2 consumed development prefix mismatch")
    return identity


def check_profile_receipt(receipt: dict, options: dict) -> None:
    validate_profile(options.get("profile"), options.get("credit_diagnostics"), options.get("mode"))
    data = receipt.get("data", {})
    prefix = data.get("evaluation_prefix", {})
    if options["profile"] == "t2-credit":
        if (any(type(data.get(key)) is not int for key in ("train_bytes", "valid_bytes", "evaluation_examples")) or
                any(type(prefix.get(key)) is not int for key in ("offset", "byte_count")) or
                data.get("train_bytes") != 671 or data.get("train_identity") != "62ac6e481ad030b9" or
                data.get("valid_bytes") != 278 or data.get("valid_identity") != "dcae5afcc74a9d72" or
                data.get("evaluation_examples") != 64 or prefix != {"offset": 0, "byte_count": 72, "identity_fnv1a64": T2_PREFIX_FNV}):
            raise ValueError("T2 numerical development identity mismatch")
