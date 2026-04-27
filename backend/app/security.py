"""Security helpers for password hashing and admin tokens."""

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional


_DEFAULT_ITERATIONS = 120000


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    if salt is None:
        salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _DEFAULT_ITERATIONS,
    )
    return "pbkdf2_sha256${}${}${}".format(
        _DEFAULT_ITERATIONS,
        _b64url_encode(salt),
        _b64url_encode(digest),
    )


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algo, iterations, salt_b64, digest_b64 = stored_hash.split("$", 3)
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    try:
        iter_count = int(iterations)
    except ValueError:
        return False
    salt = _b64url_decode(salt_b64)
    expected = _b64url_decode(digest_b64)
    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iter_count,
    )
    return hmac.compare_digest(actual, expected)


def create_admin_token(username: str, secret: str, ttl_seconds: int) -> str:
    payload = {
        "sub": username,
        "exp": int(time.time()) + ttl_seconds,
    }
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), payload_json, hashlib.sha256).digest()
    return "{}.{}".format(_b64url_encode(payload_json), _b64url_encode(signature))


def verify_admin_token(token: str, secret: str) -> Optional[dict]:
    try:
        payload_b64, signature_b64 = token.split(".", 1)
        payload_json = _b64url_decode(payload_b64)
        signature = _b64url_decode(signature_b64)
    except Exception:
        return None

    expected_sig = hmac.new(secret.encode("utf-8"), payload_json, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected_sig):
        return None

    try:
        payload = json.loads(payload_json.decode("utf-8"))
    except Exception:
        return None

    if payload.get("exp") is None or int(payload["exp"]) < int(time.time()):
        return None

    return payload
