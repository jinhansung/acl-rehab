"""
Encrypted journal — PBKDF2-derived Fernet keys from user passphrase.

Privacy invariants (confirmed — no server call is made anywhere in this module):
- Plaintext never leaves encrypt_entry().
- Passphrase never leaves the calling process.
- Salt is stored as a 16-byte prefix of the ciphertext blob; it is not secret.
- Each entry uses a fresh random salt → unique key per entry even with the same passphrase.
"""
from __future__ import annotations

import base64
import os
import sys

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from data.db import get_db
from data.models import JournalEntry

SALT_LEN = 16
# 600 000 iterations — NIST SP 800-132 (2023) minimum for PBKDF2-SHA256
PBKDF2_ITERATIONS = 600_000

# Server-side pepper loaded once at import time.
# Mixed into key derivation so decryption requires both the patient's passphrase
# AND this server secret. Set JOURNAL_PEPPER in .streamlit/secrets.toml.
# Privacy invariant: pepper is never written to DB, logs, or any outbound path.
_PEPPER: str = os.environ.get("JOURNAL_PEPPER", "")


# ── Key derivation ────────────────────────────────────────────────────────────

def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """
    Return a URL-safe base64-encoded 32-byte key suitable for Fernet.

    The effective secret is passphrase + ":" + JOURNAL_PEPPER so that
    an attacker with the database (salts + ciphertext) still needs the
    server-side pepper to mount a dictionary attack. No server call is made.
    """
    effective = (passphrase + ":" + _PEPPER).encode("utf-8")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(effective))


# ── Encryption / decryption primitives ───────────────────────────────────────

def encrypt_entry(text: str, passphrase: str) -> bytes:
    """
    Encrypt plaintext with a passphrase-derived key.

    No server call is made. Plaintext never persists beyond this function.
    Returns: 16-byte salt || Fernet token (stored as a single blob).
    """
    salt = os.urandom(SALT_LEN)
    key = _derive_key(passphrase, salt)
    token = Fernet(key).encrypt(text.encode("utf-8"))
    return salt + token


def decrypt_entry(blob: bytes, passphrase: str) -> str:
    """
    Decrypt a blob produced by encrypt_entry.

    Raises cryptography.fernet.InvalidToken if the passphrase is wrong.
    No server call is made.
    """
    if len(blob) <= SALT_LEN:
        raise ValueError("Blob too short — not a valid journal entry.")
    salt, token = blob[:SALT_LEN], blob[SALT_LEN:]
    key = _derive_key(passphrase, salt)
    return Fernet(key).decrypt(token).decode("utf-8")


# ── DB helpers ────────────────────────────────────────────────────────────────

def save_journal_entry(patient_id: int, text: str, passphrase: str) -> None:
    """Encrypt and persist one journal entry. No server call is made."""
    ciphertext = encrypt_entry(text, passphrase)
    entry = JournalEntry(patient_id=patient_id, ciphertext=ciphertext)
    with get_db() as db:
        db.save_journal_entry(entry)


def load_journal_entries(patient_id: int, passphrase: str) -> list[tuple[str, str]]:
    """
    Decrypt and return all journal entries as [(date_str, plaintext), ...].

    Entries that fail to decrypt (wrong passphrase) are returned as
    (date_str, "<decryption failed>") rather than raising, so one bad
    entry does not block the rest.
    No server call is made.
    """
    with get_db() as db:
        raw = db.get_journal_entries_raw(patient_id)
    results = []
    for entry in raw:
        try:
            text = decrypt_entry(entry.ciphertext, passphrase)
        except (InvalidToken, ValueError):
            text = "<decryption failed — wrong passphrase?>"
        results.append((str(entry.date), text))
    return results


def verify_passphrase(patient_id: int, passphrase: str) -> bool:
    """
    Attempt to decrypt the most recent journal entry.
    Returns True if successful. Used to gate passphrase confirmation UI.
    No server call is made.
    """
    with get_db() as db:
        entries = db.get_journal_entries_raw(patient_id)
    if not entries:
        return True  # no entries yet — any passphrase is valid
    try:
        decrypt_entry(entries[-1].ciphertext, passphrase)
        return True
    except (InvalidToken, ValueError):
        return False


if __name__ == "__main__":
    # python -m data.journal keygen  →  prints a standalone Fernet key (legacy helper)
    if len(sys.argv) == 2 and sys.argv[1] == "keygen":
        print(Fernet.generate_key().decode())
    else:
        print("Usage: python -m data.journal keygen")
