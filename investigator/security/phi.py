"""PHI masking and role-gated rehydration."""
from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass, field
from typing import Any

PHI_FIELDS: dict[str, str] = {
    "name": "PATIENT", "dob": "DOB", "mrn": "MRN", "ssn_last4": "SSN", "phone": "PHONE",
    "email": "EMAIL", "address": "ADDR", "npi": "NPI",
}

# token types each role can unmask
ROLE_CLEARANCE: dict[str, set[str]] = {
    "analyst": set(),
    "investigator": {"PATIENT", "MRN", "DOB"},
    "siu_lead": {"PATIENT", "MRN", "DOB", "PHONE", "EMAIL", "ADDR", "SSN", "NPI"},
    "auditor": {"MRN"},
}

# order matters, only the capture group gets tokenized
_FREE_TEXT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"\b\d{3}-\d{3}-\d{4}\b")),     # US format only
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    # only dates labelled as DOB
    ("DOB", re.compile(r"\b(?:DOB|born)\s*:?\s*(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)),
    ("MRN", re.compile(r"\bMRN\d{6,}\b")),
    ("PATIENT", re.compile(r"\b(?:Patient|Member|Pt\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)")),
]
# matches Vault.token_for format
_TOKEN = re.compile(r"\[([A-Z]+)_([0-9a-f]{6})\]")


@dataclass
class AuditEvent:
    actor_role: str
    action: str
    detail: str


@dataclass
class Vault:
    """Server-side token store."""
    # TODO: secret from env/KMS
    secret: bytes = b"demo-secret-rotate-me"
    _tokens: dict[str, str] = field(default_factory=dict)
    audit: list[AuditEvent] = field(default_factory=list)

    def token_for(self, kind: str, value: str) -> str:
        # hmac so the same member gets the same token everywhere
        digest = hmac.new(self.secret, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()[:6]
        tok = f"[{kind}_{digest}]"
        # 6 hex chars is fine at this size
        self._tokens[tok] = value
        return tok

    def resolve(self, token: str, role: str) -> str:
        kind = _TOKEN.match(token).group(1) if _TOKEN.match(token) else ""
        allowed = kind in ROLE_CLEARANCE.get(role, set())
        self.audit.append(AuditEvent(role, "resolve" if allowed else "resolve_denied", token))
        return self._tokens.get(token, token) if allowed else token

    def rehydrate(self, obj: Any, role: str) -> Any:
        if isinstance(obj, str):
            return _TOKEN.sub(lambda m: self.resolve(m.group(0), role), obj)
        if isinstance(obj, dict):
            return {k: self.rehydrate(v, role) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.rehydrate(v, role) for v in obj]
        return obj

    def contains_raw_phi(self, text: str, known_values: list[str]) -> list[str]:
        leaks = [v for v in known_values if v and v in text]
        for kind, pat in _FREE_TEXT_PATTERNS:
            if pat.search(text):
                leaks.append(f"pattern:{kind}")
        return leaks


class Deidentifier:
    def __init__(self, vault: Vault | None = None, known_names: list[str] | None = None):
        self.vault = vault or Vault()
        # longest first, avoids partial replacements
        self.known_names = sorted(set(known_names or []), key=len, reverse=True)

    def mask_record(self, record: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in record.items():
            if k in PHI_FIELDS and v:
                out[k] = self.vault.token_for(PHI_FIELDS[k], str(v))
            elif isinstance(v, str):
                out[k] = self.mask_text(v)
            elif isinstance(v, dict):
                out[k] = self.mask_record(v)
            elif isinstance(v, list):
                out[k] = [self.mask_record(x) if isinstance(x, dict) else (self.mask_text(x) if isinstance(x, str) else x) for x in v]
            else:
                out[k] = v
        return out

    def mask_text(self, text: str) -> str:
        if not text:
            return text
        for kind, pat in _FREE_TEXT_PATTERNS:
            def _sub(m, kind=kind):   # bind kind now
                raw = m.group(1) if m.groups() else m.group(0)
                tok = self.vault.token_for(kind, raw)
                return m.group(0).replace(raw, tok)
            text = pat.sub(_sub, text)
        for name in self.known_names:
            if name in text:
                text = text.replace(name, self.vault.token_for("PATIENT", name))
        return text
