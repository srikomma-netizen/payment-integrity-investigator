"""PHI/PII masking before anything reaches the model, and role-gated rehydration after.

Flow: record -> Deidentifier.mask_record -> tokens like [PATIENT_3f9a] -> model/tools;
model output -> Vault.rehydrate(obj, role) -> UI. The token map never leaves the server.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass, field
from typing import Any

# Which structured fields are PHI and what token prefix they map to.
PHI_FIELDS: dict[str, str] = {
    "name": "PATIENT", "dob": "DOB", "mrn": "MRN", "ssn_last4": "SSN", "phone": "PHONE",
    "email": "EMAIL", "address": "ADDR", "npi": "NPI",
}

# Role clearance: which token types a role may see re-identified.
# Anything not listed stays masked, and unknown roles get nothing.
ROLE_CLEARANCE: dict[str, set[str]] = {
    "analyst": set(),                                      # works fully on tokens
    "investigator": {"PATIENT", "MRN", "DOB"},             # needs identity to work the case
    "siu_lead": {"PATIENT", "MRN", "DOB", "PHONE", "EMAIL", "ADDR", "SSN", "NPI"},
    "auditor": {"MRN"},
}

# Order matters: patterns run in sequence and later ones see earlier tokens, not raw text.
# Where a pattern has a capture group, only the group is tokenized ("DOB 1972-..." keeps "DOB").
_FREE_TEXT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"\b\d{3}-\d{3}-\d{4}\b")),     # US 3-3-4 only; good enough for the synthetic notes
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    # bare dates are everywhere (service dates), so only mask ones introduced as a DOB
    ("DOB", re.compile(r"\b(?:DOB|born)\s*:?\s*(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)),
    ("MRN", re.compile(r"\bMRN\d{6,}\b")),
    # "Patient Maya Okafor", "member Luis Costa": names introduced by a role word in free text
    ("PATIENT", re.compile(r"\b(?:Patient|Member|Pt\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)")),
]
# Must stay in sync with the format produced by Vault.token_for.
_TOKEN = re.compile(r"\[([A-Z]+)_([0-9a-f]{6})\]")


@dataclass
class AuditEvent:
    actor_role: str
    action: str
    detail: str


@dataclass
class Vault:
    """Server-side token store. Should never be co-located with model logs."""
    # TODO: load the secret from a KMS / env var and back _tokens with an encrypted store
    secret: bytes = b"demo-secret-rotate-me"
    _tokens: dict[str, str] = field(default_factory=dict)
    audit: list[AuditEvent] = field(default_factory=list)

    def token_for(self, kind: str, value: str) -> str:
        # HMAC, not a random token: same member -> same token across claims, so the model can still link them.
        # kind is in the message so an MRN and a name with the same text don't collide.
        digest = hmac.new(self.secret, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()[:6]
        tok = f"[{kind}_{digest}]"
        # 6 hex chars is plenty at this scale; widen it if the vault ever holds millions of values
        self._tokens[tok] = value
        return tok

    def resolve(self, token: str, role: str) -> str:
        kind = _TOKEN.match(token).group(1) if _TOKEN.match(token) else ""
        allowed = kind in ROLE_CLEARANCE.get(role, set())
        # audit denials too; repeated denied lookups are the interesting ones
        self.audit.append(AuditEvent(role, "resolve" if allowed else "resolve_denied", token))
        # an unknown token (e.g. one the model made up) comes back unchanged rather than raising
        return self._tokens.get(token, token) if allowed else token

    def rehydrate(self, obj: Any, role: str) -> Any:
        """Replace tokens the role is cleared for, anywhere in a nested structure."""
        if isinstance(obj, str):
            return _TOKEN.sub(lambda m: self.resolve(m.group(0), role), obj)
        if isinstance(obj, dict):
            return {k: self.rehydrate(v, role) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.rehydrate(v, role) for v in obj]
        return obj

    def contains_raw_phi(self, text: str, known_values: list[str]) -> list[str]:
        """Leak check used by tests and the verify node."""
        leaks = [v for v in known_values if v and v in text]
        # also catch PHI-shaped strings we were never told about
        for kind, pat in _FREE_TEXT_PATTERNS:
            if pat.search(text):
                leaks.append(f"pattern:{kind}")
        return leaks


class Deidentifier:
    def __init__(self, vault: Vault | None = None, known_names: list[str] | None = None):
        self.vault = vault or Vault()
        # longest first so "Maya Okafor" is replaced before a shorter name that's a substring of it
        self.known_names = sorted(set(known_names or []), key=len, reverse=True)

    def mask_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Tokenize known PHI fields by key, and scan every other string for PHI in free text."""
        out: dict[str, Any] = {}
        for k, v in record.items():
            # keyed on field name, so "name" is masked even on provider records; harmless and safer
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
            # kind=kind binds the loop variable now; a plain closure would see the last kind only
            def _sub(m, kind=kind):
                raw = m.group(1) if m.groups() else m.group(0)
                tok = self.vault.token_for(kind, raw)
                return m.group(0).replace(raw, tok)
            text = pat.sub(_sub, text)
        # regexes miss names without a "Patient"/"Member" prefix, so sweep the roster as a backstop
        for name in self.known_names:
            if name in text:
                text = text.replace(name, self.vault.token_for("PATIENT", name))
        return text
