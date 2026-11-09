"""Waveshare certificate and bundle validation."""

import ipaddress
from datetime import datetime, timedelta, timezone

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from scripts.create_local_ca import issue_certificates
from server import satellite_bundle, satellite_waveshare


def _certificate(names, *, expired=False):
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "device.example")])
    now = datetime.now(timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=2))
        .not_valid_after(now - timedelta(days=1) if expired else now + timedelta(days=1))
    )
    if names is not None:
        builder = builder.add_extension(x509.SubjectAlternativeName(names), critical=False)
    cert = builder.sign(key, hashes.SHA256())
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def _bundle(host, cert):
    return (
        {
            "schema_version": 1,
            "device": {"id": "test", "name": "Test", "timezone": "UTC0"},
            "server": {"url": f"wss://{host}/ws/satellite-v2", "certificate_pem": cert},
            "audio": {"speaker_volume_percent": 50, "aec_enabled": True},
            "music": {"player": "none"},
        },
        {
            "schema_version": 1,
            "wifi": {"ssid": "network", "password": "secret"},
            "fulloch": {"device_token": "token"},
        },
    )


def test_certificate_hostname_and_ip_matching():
    cert = _certificate(
        [x509.DNSName("device.example"), x509.IPAddress(ipaddress.ip_address("192.0.2.1"))]
    )
    satellite_waveshare.validate_bundle(*_bundle("device.example", cert))
    satellite_waveshare.validate_bundle(*_bundle("192.0.2.1", cert))
    with pytest.raises(satellite_bundle.BundleError, match="does not match"):
        satellite_waveshare.validate_bundle(*_bundle("wrong.example", cert))
    with pytest.raises(satellite_bundle.BundleError, match="does not match"):
        satellite_waveshare.validate_bundle(*_bundle("192.0.2.2", cert))


def test_rejects_expired_certificates_and_private_keys():
    cert = _certificate([x509.DNSName("device.example")], expired=True)
    with pytest.raises(satellite_bundle.BundleError, match="expired"):
        satellite_waveshare.validate_bundle(*_bundle("device.example", cert))
    with pytest.raises(satellite_bundle.BundleError, match="public PEM"):
        satellite_waveshare.validate_bundle(*_bundle("device.example", cert + "PRIVATE KEY"))


def test_dns_wildcard_and_san_precedence_are_strict(monkeypatch):
    import ssl

    # Python 3.12 removed ssl.match_hostname; validation must not rely on it.
    monkeypatch.delattr(ssl, "match_hostname", raising=False)
    wildcard = _certificate([x509.DNSName("*.example.com")])
    satellite_waveshare.validate_bundle(*_bundle("ROOM.example.com", wildcard))
    for host in ("example.com", "a.b.example.com", "device.example"):
        with pytest.raises(satellite_bundle.BundleError, match="does not match"):
            satellite_waveshare.validate_bundle(*_bundle(host, wildcard))
    partial = _certificate([x509.DNSName("d*.example.com")])
    with pytest.raises(satellite_bundle.BundleError, match="does not match"):
        satellite_waveshare.validate_bundle(*_bundle("desk.example.com", partial))
    # The common name is device.example, but SAN takes precedence.
    with pytest.raises(satellite_bundle.BundleError, match="does not match"):
        satellite_waveshare.validate_bundle(*_bundle("device.example", partial))


def test_ipv6_address_must_match_ip_subject_alternative_name():
    cert = _certificate([x509.IPAddress(ipaddress.ip_address("2001:db8::1"))])
    satellite_waveshare.validate_bundle(*_bundle("[2001:db8::1]", cert))
    with pytest.raises(satellite_bundle.BundleError, match="does not match"):
        satellite_waveshare.validate_bundle(*_bundle("[2001:db8::2]", cert))


def test_common_name_fallback_is_limited_to_dns_without_san():
    cert = _certificate(None)
    satellite_waveshare.validate_bundle(*_bundle("device.example", cert))
    with pytest.raises(satellite_bundle.BundleError, match="does not match"):
        satellite_waveshare.validate_bundle(*_bundle("other.example", cert))
    with pytest.raises(satellite_bundle.BundleError, match="IP address subject alternative name"):
        satellite_waveshare.validate_bundle(*_bundle("192.0.2.1", cert))


def test_generates_real_nvs_with_installed_generator():
    cert = _certificate([x509.DNSName("device.example")])
    image = satellite_waveshare.generate_nvs(*_bundle("device.example", cert))
    assert len(image) == 0x10000


def test_dashboard_ca_is_verified_and_leaf_identity_is_checked(tmp_path):
    ca_path, leaf_path, _ = issue_certificates(tmp_path, ["device.example"], [], False)
    leaf = leaf_path.read_text()
    trust = satellite_waveshare.dashboard_trust_certificate(leaf, ca_path)
    assert trust == ca_path.read_text()
    config, credentials = _bundle("device.example", trust)
    satellite_waveshare.validate_bundle(config, credentials, identity_certificate=leaf)
    assert (
        len(satellite_waveshare.generate_nvs(config, credentials, identity_certificate=leaf))
        == 0x10000
    )
    config["server"]["url"] = "wss://wrong.example/ws/satellite-v2"
    with pytest.raises(satellite_bundle.BundleError, match="does not match"):
        satellite_waveshare.validate_bundle(config, credentials, identity_certificate=leaf)


def test_dashboard_certificate_rejects_missing_or_unrelated_ca(tmp_path):
    _, leaf_path, _ = issue_certificates(tmp_path / "one", ["device.example"], [], False)
    other_ca, _, _ = issue_certificates(tmp_path / "two", ["device.example"], [], False)
    leaf = leaf_path.read_text()
    for ca in (None, other_ca, tmp_path / "missing"):
        with pytest.raises(satellite_bundle.BundleError, match="local CA"):
            satellite_waveshare.dashboard_trust_certificate(leaf, ca)


def test_self_signed_dashboard_ignores_stale_ca(tmp_path):
    from core.tls_certs import ensure_self_signed_cert

    cert_path, _ = ensure_self_signed_cert(str(tmp_path / "dashboard"))
    other_ca, _, _ = issue_certificates(tmp_path / "other", [], [], False)
    cert = open(cert_path, encoding="utf-8").read()
    assert satellite_waveshare.dashboard_trust_certificate(cert, other_ca) == cert
