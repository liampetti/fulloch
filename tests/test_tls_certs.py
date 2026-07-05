"""Self-signed dashboard HTTPS certificate generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tls_certs import ensure_self_signed_cert, regenerate_self_signed_cert  # noqa: E402


def test_generates_cert_and_key(tmp_path):
    cert_path, key_path = ensure_self_signed_cert(str(tmp_path / "certs"))
    assert Path(cert_path).is_file()
    assert Path(key_path).is_file()
    assert Path(cert_path).read_bytes().startswith(b"-----BEGIN CERTIFICATE-----")
    assert b"PRIVATE KEY" in Path(key_path).read_bytes()


def test_cert_covers_localhost_and_loopback(tmp_path):
    from cryptography import x509

    cert_path, _ = ensure_self_signed_cert(str(tmp_path / "certs"))
    cert = x509.load_pem_x509_certificate(Path(cert_path).read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "localhost" in san.get_values_for_type(x509.DNSName)
    ips = [str(ip) for ip in san.get_values_for_type(x509.IPAddress)]
    assert "127.0.0.1" in ips


def test_idempotent_does_not_regenerate(tmp_path):
    certs_dir = tmp_path / "certs"
    cert_path, key_path = ensure_self_signed_cert(str(certs_dir))
    original = Path(cert_path).read_bytes()

    cert_path2, key_path2 = ensure_self_signed_cert(str(certs_dir))
    assert cert_path == cert_path2
    assert key_path == key_path2
    assert Path(cert_path2).read_bytes() == original


def test_leaves_existing_user_supplied_cert_untouched(tmp_path):
    certs_dir = tmp_path / "certs"
    certs_dir.mkdir()
    (certs_dir / "dashboard.crt").write_text("MY CERT")
    (certs_dir / "dashboard.key").write_text("MY KEY")

    cert_path, key_path = ensure_self_signed_cert(str(certs_dir))
    assert Path(cert_path).read_text() == "MY CERT"
    assert Path(key_path).read_text() == "MY KEY"


def test_regenerate_overwrites_existing_pair(tmp_path):
    certs_dir = tmp_path / "certs"
    cert_path, key_path = ensure_self_signed_cert(str(certs_dir))
    original = Path(cert_path).read_bytes()

    cert_path2, key_path2 = regenerate_self_signed_cert(str(certs_dir))
    assert cert_path == cert_path2
    assert key_path == key_path2
    assert Path(cert_path2).read_bytes() != original
    assert Path(cert_path2).read_bytes().startswith(b"-----BEGIN CERTIFICATE-----")


def test_regenerate_overwrites_user_supplied_cert(tmp_path):
    certs_dir = tmp_path / "certs"
    certs_dir.mkdir()
    (certs_dir / "dashboard.crt").write_text("MY CERT")
    (certs_dir / "dashboard.key").write_text("MY KEY")

    cert_path, key_path = regenerate_self_signed_cert(str(certs_dir))
    assert Path(cert_path).read_bytes().startswith(b"-----BEGIN CERTIFICATE-----")
    assert b"PRIVATE KEY" in Path(key_path).read_bytes()
