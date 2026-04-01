"""AES-256-GCM encryption with PBKDF2 key derivation.

Every piece of user context stored in Kangaroo Shift is encrypted at rest.
We use AES-256-GCM because:
    - AES-256: NIST-approved, quantum-resistant key length
    - GCM mode: authenticated encryption (integrity + confidentiality)
    - Each encryption uses a unique nonce (no IV reuse)

Key derivation:
    - PBKDF2-HMAC-SHA256 with 600,000 iterations (OWASP 2024 recommendation)
    - 32-byte random salt per key derivation
    - Master key derived from user password or system secret

This module handles raw byte-level encryption. The Vault (storage/vault.py)
wraps this with JSON serialization and database storage.

Usage:
    from app.core.security.encryption import EncryptionEngine

    engine = EncryptionEngine(master_key="your-secret-key")
    encrypted = engine.encrypt(b"sensitive data")
    decrypted = engine.decrypt(encrypted)
    assert decrypted == b"sensitive data"
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


# -- Constants ---------------------------------------------------------------

_KEY_LENGTH = 32       # 256 bits for AES-256
_NONCE_LENGTH = 12     # 96 bits (standard for GCM)
_SALT_LENGTH = 32      # 256 bits for key derivation salt
_KDF_ITERATIONS = 600_000  # OWASP 2024 recommendation for PBKDF2-SHA256
_TAG_LENGTH = 16       # 128-bit authentication tag (GCM default)


# -- Encrypted Payload -------------------------------------------------------


@dataclass(frozen=True)
class EncryptedPayload:
    """An encrypted blob with all metadata needed for decryption.

    Fields:
        ciphertext: The encrypted data (includes GCM auth tag).
        nonce: Unique per encryption, needed for decryption.
        salt: Used for key derivation (if key was derived from password).
        version: Encryption scheme version for forward compatibility.
    """

    ciphertext: bytes
    nonce: bytes
    salt: bytes
    version: int = 1

    def to_bytes(self) -> bytes:
        """Serialize to a single byte string for storage.

        Format: [version:1][salt:32][nonce:12][ciphertext:N]
        """
        return (
            self.version.to_bytes(1, "big")
            + self.salt
            + self.nonce
            + self.ciphertext
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "EncryptedPayload":
        """Deserialize from a byte string.

        Args:
            data: Serialized encrypted payload.

        Returns:
            EncryptedPayload instance.

        Raises:
            ValueError: If data is too short or version is unsupported.
        """
        min_length = 1 + _SALT_LENGTH + _NONCE_LENGTH + _TAG_LENGTH
        if len(data) < min_length:
            raise ValueError(
                f"Encrypted payload too short: {len(data)} bytes "
                f"(minimum {min_length})"
            )

        version = data[0]
        if version != 1:
            raise ValueError(f"Unsupported encryption version: {version}")

        offset = 1
        salt = data[offset:offset + _SALT_LENGTH]
        offset += _SALT_LENGTH
        nonce = data[offset:offset + _NONCE_LENGTH]
        offset += _NONCE_LENGTH
        ciphertext = data[offset:]

        return cls(
            ciphertext=ciphertext,
            nonce=nonce,
            salt=salt,
            version=version,
        )


# -- Key Derivation ----------------------------------------------------------


def derive_key(
    secret: str | bytes,
    salt: bytes | None = None,
    iterations: int = _KDF_ITERATIONS,
) -> tuple[bytes, bytes]:
    """Derive an AES-256 key from a secret using PBKDF2.

    Args:
        secret: Master secret (password or system key).
        salt: Random salt. Generated if not provided.
        iterations: PBKDF2 iteration count.

    Returns:
        Tuple of (derived_key, salt).
    """
    if salt is None:
        salt = os.urandom(_SALT_LENGTH)

    if isinstance(secret, str):
        secret = secret.encode("utf-8")

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=_KEY_LENGTH,
        salt=salt,
        iterations=iterations,
    )
    key = kdf.derive(secret)
    return key, salt


# -- Encryption Engine -------------------------------------------------------


class EncryptionEngine:
    """AES-256-GCM encryption engine.

    Thread-safe: each encrypt() call generates a unique nonce.
    No state shared between operations.

    Usage:
        engine = EncryptionEngine(master_key="system-secret")
        encrypted = engine.encrypt(b"sensitive data")
        decrypted = engine.decrypt(encrypted)
    """

    def __init__(
        self,
        master_key: str | bytes,
        kdf_iterations: int = _KDF_ITERATIONS,
    ) -> None:
        """Initialize with a master key.

        Args:
            master_key: System secret for key derivation.
            kdf_iterations: PBKDF2 iterations (higher = slower but safer).
        """
        self._master_key = (
            master_key if isinstance(master_key, bytes)
            else master_key.encode("utf-8")
        )
        self._kdf_iterations = kdf_iterations

    def encrypt(
        self,
        plaintext: bytes,
        associated_data: bytes | None = None,
    ) -> EncryptedPayload:
        """Encrypt data using AES-256-GCM.

        Each call generates a unique salt and nonce, so encrypting
        the same plaintext twice produces different ciphertext.

        Args:
            plaintext: Data to encrypt.
            associated_data: Optional AAD for authenticated encryption.
                             Not encrypted, but integrity-protected.

        Returns:
            EncryptedPayload with ciphertext, nonce, and salt.
        """
        # Derive a unique key for this encryption
        key, salt = derive_key(self._master_key, iterations=self._kdf_iterations)

        # Generate unique nonce
        nonce = os.urandom(_NONCE_LENGTH)

        # Encrypt
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)

        return EncryptedPayload(
            ciphertext=ciphertext,
            nonce=nonce,
            salt=salt,
        )

    def decrypt(
        self,
        payload: EncryptedPayload,
        associated_data: bytes | None = None,
    ) -> bytes:
        """Decrypt an encrypted payload.

        Args:
            payload: EncryptedPayload from encrypt().
            associated_data: Must match what was passed to encrypt().

        Returns:
            Original plaintext bytes.

        Raises:
            ValueError: If decryption fails (wrong key, tampered data).
        """
        # Re-derive the key using the stored salt
        key, _ = derive_key(
            self._master_key,
            salt=payload.salt,
            iterations=self._kdf_iterations,
        )

        aesgcm = AESGCM(key)
        try:
            plaintext = aesgcm.decrypt(payload.nonce, payload.ciphertext, associated_data)
        except Exception as e:
            raise ValueError(
                "Decryption failed. Wrong key or tampered data."
            ) from e

        return plaintext

    def encrypt_string(
        self,
        text: str,
        associated_data: bytes | None = None,
    ) -> EncryptedPayload:
        """Convenience: encrypt a UTF-8 string."""
        return self.encrypt(text.encode("utf-8"), associated_data)

    def decrypt_string(
        self,
        payload: EncryptedPayload,
        associated_data: bytes | None = None,
    ) -> str:
        """Convenience: decrypt to a UTF-8 string."""
        return self.decrypt(payload, associated_data).decode("utf-8")

    @staticmethod
    def generate_key() -> str:
        """Generate a cryptographically secure random key.

        Use this to generate master keys for production.
        Returns a 64-character hex string (256 bits).
        """
        return secrets.token_hex(32)
