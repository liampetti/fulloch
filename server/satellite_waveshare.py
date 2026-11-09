"""Waveshare Rev 2 bundle validation, independent of deployment and transactions."""

import ipaddress
from datetime import datetime, timezone
from urllib.parse import urlsplit

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.x509.oid import ExtensionOID, NameOID

from . import satellite_bundle


def _dns_matches(pattern: str, host: str) -> bool:
    """Match exact DNS names or a whole leftmost wildcard, as on the device."""
    pattern, host = pattern.lower(), host.lower()
    if not pattern.isascii() or not host.isascii() or "\x00" in pattern or "\x00" in host:
        return False
    if "*" in host:
        return False
    if "*" not in pattern:
        return pattern == host
    if not pattern.startswith("*.") or pattern.count("*") != 1:
        return False
    first, separator, suffix = host.partition(".")
    return bool(first and separator and suffix and suffix == pattern[2:])


def dashboard_trust_certificate(certificate_pem: str, ca_path) -> str:
    """Select the dashboard's self-signed certificate or its local CA.

    Only use a CA that actually signed the active leaf, never a stale CA file.
    """
    try:
        leaf = x509.load_pem_x509_certificate(certificate_pem.encode("utf-8"))
        if leaf.issuer == leaf.subject:
            leaf.verify_directly_issued_by(leaf)
            return certificate_pem
        if ca_path is None:
            raise ValueError("missing CA")
        ca_pem = ca_path.read_text(encoding="utf-8")
        ca = x509.load_pem_x509_certificate(ca_pem.encode("utf-8"))
        now = datetime.now(timezone.utc)
        if not ca.not_valid_before_utc <= now <= ca.not_valid_after_utc:
            raise ValueError("CA expired")
        if not ca.extensions.get_extension_for_class(x509.BasicConstraints).value.ca:
            raise ValueError("not a CA")
        try:
            if not ca.extensions.get_extension_for_class(x509.KeyUsage).value.key_cert_sign:
                raise ValueError("CA cannot sign certificates")
        except x509.ExtensionNotFound:
            pass
        leaf.verify_directly_issued_by(ca)
        return ca_pem
    except (ValueError, TypeError, OSError, InvalidSignature, x509.ExtensionNotFound) as error:
        raise satellite_bundle.BundleError(
            "Cannot verify the dashboard certificate with its local CA; check the public CA certificate"
        ) from error


def validate_bundle(
    config: dict, credentials: dict, *, identity_certificate: str | None = None
) -> None:
    """Validate the firmware bundle and the public certificate's server identity."""
    satellite_bundle.validate(config, credentials)
    server = config["server"]
    try:
        host = urlsplit(server["url"]).hostname
        certificate = x509.load_pem_x509_certificate(
            (identity_certificate or server["certificate_pem"]).encode("utf-8")
        )
        now = datetime.now(timezone.utc)
        if not certificate.not_valid_before_utc <= now <= certificate.not_valid_after_utc:
            raise satellite_bundle.BundleError("server certificate is expired or not yet valid")
        try:
            san = certificate.extensions.get_extension_for_oid(
                ExtensionOID.SUBJECT_ALTERNATIVE_NAME
            ).value
        except x509.ExtensionNotFound:
            san = None
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None:
            if san is None:
                raise satellite_bundle.BundleError(
                    "server certificate needs an IP address subject alternative name"
                )
            matched = address in san.get_values_for_type(x509.IPAddress)
        else:
            # Match mbedTLS: use a common name only when there is no SAN extension.
            names = (
                san.get_values_for_type(x509.DNSName)
                if san is not None
                else [
                    entry.value
                    for entry in certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
                ]
            )
            matched = any(_dns_matches(name, host) for name in names)
        if not matched:
            raise ValueError("hostname mismatch")
    except satellite_bundle.BundleError:
        raise
    except (ValueError, TypeError) as error:
        raise satellite_bundle.BundleError(
            "server certificate does not match the selected server host"
        ) from error


def generate_nvs(
    config: dict, credentials: dict, *, identity_certificate: str | None = None
) -> bytes:
    """Generate a compatible private NVS image after profile validation."""
    validate_bundle(config, credentials, identity_certificate=identity_certificate)
    return satellite_bundle.generate_nvs(config, credentials)
