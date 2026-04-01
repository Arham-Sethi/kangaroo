"""Tests for AES-256-GCM encryption engine.

Tests cover:
    - Key derivation (PBKDF2-HMAC-SHA256)
    - Encrypt/decrypt round-trips
    - Unique nonces per encryption
    - Associated data (AAD) binding
    - Payload serialization/deserialization
    - Error handling (wrong key, tampered data, short payload)
    - String convenience methods
    - Key generation
"""

import os

import pytest

from app.core.security.encryption import (
    EncryptedPayload,
    EncryptionEngine,
    _KEY_LENGTH,
    _KDF_ITERATIONS,
    _NONCE_LENGTH,
    _SALT_LENGTH,
    _TAG_LENGTH,
    derive_key,
)


# == Key Derivation ==========================================================


class TestDeriveKey:
    """Tests for PBKDF2 key derivation."""

    def test_derive_key_returns_correct_length(self) -> None:
        key, salt = derive_key("test-secret")
        assert len(key) == _KEY_LENGTH
        assert len(salt) == _SALT_LENGTH

    def test_derive_key_generates_unique_salt(self) -> None:
        _, salt1 = derive_key("same-secret")
        _, salt2 = derive_key("same-secret")
        assert salt1 != salt2

    def test_derive_key_with_provided_salt_is_deterministic(self) -> None:
        fixed_salt = os.urandom(_SALT_LENGTH)
        key1, _ = derive_key("my-secret", salt=fixed_salt)
        key2, _ = derive_key("my-secret", salt=fixed_salt)
        assert key1 == key2

    def test_derive_key_different_secrets_produce_different_keys(self) -> None:
        fixed_salt = os.urandom(_SALT_LENGTH)
        key1, _ = derive_key("secret-a", salt=fixed_salt)
        key2, _ = derive_key("secret-b", salt=fixed_salt)
        assert key1 != key2

    def test_derive_key_accepts_bytes_secret(self) -> None:
        key, salt = derive_key(b"binary-secret")
        assert len(key) == _KEY_LENGTH

    def test_derive_key_custom_iterations(self) -> None:
        key, salt = derive_key("test", iterations=1000)
        assert len(key) == _KEY_LENGTH

    def test_derive_key_returns_provided_salt(self) -> None:
        provided_salt = os.urandom(_SALT_LENGTH)
        _, returned_salt = derive_key("test", salt=provided_salt)
        assert returned_salt == provided_salt


# == EncryptedPayload ========================================================


class TestEncryptedPayload:
    """Tests for payload serialization."""

    def test_to_bytes_format(self) -> None:
        payload = EncryptedPayload(
            ciphertext=b"encrypted-data",
            nonce=b"x" * _NONCE_LENGTH,
            salt=b"y" * _SALT_LENGTH,
            version=1,
        )
        raw = payload.to_bytes()
        assert raw[0] == 1  # version byte
        assert len(raw) == 1 + _SALT_LENGTH + _NONCE_LENGTH + len(b"encrypted-data")

    def test_round_trip_serialization(self) -> None:
        original = EncryptedPayload(
            ciphertext=b"test-cipher-with-tag-data",
            nonce=os.urandom(_NONCE_LENGTH),
            salt=os.urandom(_SALT_LENGTH),
            version=1,
        )
        raw = original.to_bytes()
        restored = EncryptedPayload.from_bytes(raw)
        assert restored.ciphertext == original.ciphertext
        assert restored.nonce == original.nonce
        assert restored.salt == original.salt
        assert restored.version == original.version

    def test_from_bytes_rejects_short_data(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            EncryptedPayload.from_bytes(b"short")

    def test_from_bytes_rejects_unsupported_version(self) -> None:
        min_length = 1 + _SALT_LENGTH + _NONCE_LENGTH + _TAG_LENGTH
        data = bytes([99]) + b"\x00" * (min_length - 1)
        with pytest.raises(ValueError, match="Unsupported encryption version"):
            EncryptedPayload.from_bytes(data)

    def test_payload_is_frozen(self) -> None:
        payload = EncryptedPayload(
            ciphertext=b"data",
            nonce=b"x" * _NONCE_LENGTH,
            salt=b"y" * _SALT_LENGTH,
        )
        with pytest.raises(AttributeError):
            payload.ciphertext = b"new"  # type: ignore[misc]


# == EncryptionEngine ========================================================


class TestEncryptionEngine:
    """Tests for the main encryption engine."""

    @pytest.fixture
    def engine(self) -> EncryptionEngine:
        return EncryptionEngine(master_key="test-key-12345", kdf_iterations=1000)

    def test_encrypt_returns_payload(self, engine: EncryptionEngine) -> None:
        payload = engine.encrypt(b"hello world")
        assert isinstance(payload, EncryptedPayload)
        assert len(payload.ciphertext) > 0
        assert len(payload.nonce) == _NONCE_LENGTH
        assert len(payload.salt) == _SALT_LENGTH

    def test_decrypt_round_trip(self, engine: EncryptionEngine) -> None:
        plaintext = b"sensitive data that must be protected"
        payload = engine.encrypt(plaintext)
        decrypted = engine.decrypt(payload)
        assert decrypted == plaintext

    def test_encrypt_produces_unique_ciphertext(self, engine: EncryptionEngine) -> None:
        plaintext = b"same data"
        p1 = engine.encrypt(plaintext)
        p2 = engine.encrypt(plaintext)
        assert p1.ciphertext != p2.ciphertext
        assert p1.nonce != p2.nonce
        assert p1.salt != p2.salt

    def test_decrypt_with_wrong_key_fails(self, engine: EncryptionEngine) -> None:
        payload = engine.encrypt(b"secret")
        wrong_engine = EncryptionEngine(master_key="wrong-key", kdf_iterations=1000)
        with pytest.raises(ValueError, match="Decryption failed"):
            wrong_engine.decrypt(payload)

    def test_encrypt_with_associated_data(self, engine: EncryptionEngine) -> None:
        plaintext = b"data"
        aad = b"session-123"
        payload = engine.encrypt(plaintext, associated_data=aad)
        decrypted = engine.decrypt(payload, associated_data=aad)
        assert decrypted == plaintext

    def test_decrypt_with_wrong_aad_fails(self, engine: EncryptionEngine) -> None:
        payload = engine.encrypt(b"data", associated_data=b"correct-aad")
        with pytest.raises(ValueError, match="Decryption failed"):
            engine.decrypt(payload, associated_data=b"wrong-aad")

    def test_decrypt_with_missing_aad_fails(self, engine: EncryptionEngine) -> None:
        payload = engine.encrypt(b"data", associated_data=b"some-aad")
        with pytest.raises(ValueError, match="Decryption failed"):
            engine.decrypt(payload, associated_data=None)

    def test_encrypt_empty_plaintext(self, engine: EncryptionEngine) -> None:
        payload = engine.encrypt(b"")
        decrypted = engine.decrypt(payload)
        assert decrypted == b""

    def test_encrypt_large_data(self, engine: EncryptionEngine) -> None:
        plaintext = os.urandom(100_000)
        payload = engine.encrypt(plaintext)
        decrypted = engine.decrypt(payload)
        assert decrypted == plaintext

    def test_full_serialization_round_trip(self, engine: EncryptionEngine) -> None:
        plaintext = b"full round trip through bytes"
        payload = engine.encrypt(plaintext)
        raw = payload.to_bytes()
        restored = EncryptedPayload.from_bytes(raw)
        decrypted = engine.decrypt(restored)
        assert decrypted == plaintext

    def test_engine_accepts_bytes_key(self) -> None:
        engine = EncryptionEngine(master_key=b"bytes-key", kdf_iterations=1000)
        payload = engine.encrypt(b"test")
        assert engine.decrypt(payload) == b"test"


# == String Convenience Methods ==============================================


class TestStringMethods:
    """Tests for encrypt_string and decrypt_string."""

    @pytest.fixture
    def engine(self) -> EncryptionEngine:
        return EncryptionEngine(master_key="string-test-key", kdf_iterations=1000)

    def test_encrypt_string_round_trip(self, engine: EncryptionEngine) -> None:
        text = "Hello, world! Unicode: \u00e9\u00e0\u00fc\u00f1"
        payload = engine.encrypt_string(text)
        decrypted = engine.decrypt_string(payload)
        assert decrypted == text

    def test_encrypt_string_with_aad(self, engine: EncryptionEngine) -> None:
        text = "secret message"
        aad = b"context-id"
        payload = engine.encrypt_string(text, associated_data=aad)
        decrypted = engine.decrypt_string(payload, associated_data=aad)
        assert decrypted == text

    def test_encrypt_empty_string(self, engine: EncryptionEngine) -> None:
        payload = engine.encrypt_string("")
        assert engine.decrypt_string(payload) == ""


# == Key Generation ==========================================================


class TestKeyGeneration:
    """Tests for static key generation."""

    def test_generate_key_length(self) -> None:
        key = EncryptionEngine.generate_key()
        assert len(key) == 64  # 32 bytes = 64 hex chars

    def test_generate_key_is_hex(self) -> None:
        key = EncryptionEngine.generate_key()
        int(key, 16)  # should not raise

    def test_generate_key_is_unique(self) -> None:
        keys = {EncryptionEngine.generate_key() for _ in range(10)}
        assert len(keys) == 10


# == Tamper Detection ========================================================


class TestTamperDetection:
    """Tests for GCM authentication tag tamper detection."""

    @pytest.fixture
    def engine(self) -> EncryptionEngine:
        return EncryptionEngine(master_key="tamper-test", kdf_iterations=1000)

    def test_tampered_ciphertext_detected(self, engine: EncryptionEngine) -> None:
        payload = engine.encrypt(b"original data")
        tampered = EncryptedPayload(
            ciphertext=payload.ciphertext[:-1] + bytes([payload.ciphertext[-1] ^ 0xFF]),
            nonce=payload.nonce,
            salt=payload.salt,
        )
        with pytest.raises(ValueError, match="Decryption failed"):
            engine.decrypt(tampered)

    def test_tampered_nonce_detected(self, engine: EncryptionEngine) -> None:
        payload = engine.encrypt(b"original data")
        tampered = EncryptedPayload(
            ciphertext=payload.ciphertext,
            nonce=os.urandom(_NONCE_LENGTH),
            salt=payload.salt,
        )
        with pytest.raises(ValueError, match="Decryption failed"):
            engine.decrypt(tampered)
