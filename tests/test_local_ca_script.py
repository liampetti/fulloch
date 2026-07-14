"""Local CA helper tests without touching the real data/certs directory."""

import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).parent.parent / "scripts" / "create_local_ca.py"
    spec = importlib.util.spec_from_file_location("create_local_ca", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_issues_dashboard_cert_signed_by_reusable_ca(tmp_path):
    from cryptography import x509

    script = _load_script()
    ca_path, cert_path, key_path = script.issue_certificates(
        tmp_path, ["fulloch.home"], ["192.168.1.20"], force=True
    )
    ca = x509.load_pem_x509_certificate(ca_path.read_bytes())
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    sans = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value

    assert key_path.is_file()
    assert cert.issuer == ca.subject
    assert "fulloch.home" in sans.get_values_for_type(x509.DNSName)
    assert "192.168.1.20" in [str(ip) for ip in sans.get_values_for_type(x509.IPAddress)]


def test_refuses_to_overwrite_dashboard_cert_without_force(tmp_path):
    import pytest

    script = _load_script()
    script.issue_certificates(tmp_path, [], [], force=True)

    with pytest.raises(RuntimeError, match="--force"):
        script.issue_certificates(tmp_path, [], [], force=False)
