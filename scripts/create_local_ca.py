#!/usr/bin/env python3
"""Create a private LAN CA and a Fulloch dashboard certificate it signs.

Run from any directory. The public CA certificate is installed once in each
browser device's trust store; never copy the CA private key off this host.
"""

import argparse
import ipaddress
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

CA_CERT_FILENAME = "fulloch-home-ca.crt"
CA_KEY_FILENAME = "fulloch-home-ca.key"
DASHBOARD_CERT_FILENAME = "dashboard.crt"
DASHBOARD_KEY_FILENAME = "dashboard.key"


def _local_ips() -> set[str]:
    ips = {"127.0.0.1", "::1"}
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ips.add(sock.getsockname()[0])
    except OSError:
        pass
    return ips


def _write_key(path: Path, key) -> None:
    from cryptography.hazmat.primitives import serialization

    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def issue_certificates(certs_dir: Path, hosts: list[str], ips: list[str], force: bool) -> tuple[Path, Path, Path]:
    """Create/reuse the CA and issue dashboard.crt/dashboard.key.

    Returns (ca_cert, dashboard_cert, dashboard_key). `force` only replaces the
    dashboard leaf certificate so already-trusted devices keep trusting the CA.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    certs_dir.mkdir(parents=True, exist_ok=True)
    ca_cert_path = certs_dir / CA_CERT_FILENAME
    ca_key_path = certs_dir / CA_KEY_FILENAME
    cert_path = certs_dir / DASHBOARD_CERT_FILENAME
    key_path = certs_dir / DASHBOARD_KEY_FILENAME

    if ca_cert_path.exists() != ca_key_path.exists():
        raise RuntimeError(f"CA files must be a pair: {ca_cert_path} and {ca_key_path}")
    if ca_cert_path.exists():
        ca_cert = x509.load_pem_x509_certificate(ca_cert_path.read_bytes())
        ca_key = serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None)
    else:
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Fulloch Home CA")])
        now = datetime.now(timezone.utc)
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=False,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(ca_key, hashes.SHA256())
        )
        ca_cert_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
        _write_key(ca_key_path, ca_key)

    if (cert_path.exists() or key_path.exists()) and not force:
        raise RuntimeError(
            f"{cert_path} or {key_path} already exists; re-run with --force to replace the dashboard certificate"
        )

    names = {"localhost", socket.gethostname(), *hosts}
    sans = [x509.DNSName(name) for name in sorted(name for name in names if name)]
    for raw_ip in sorted({*_local_ips(), *ips}):
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(raw_ip)))
        except ValueError as exc:
            raise RuntimeError(f"invalid IP address: {raw_ip}") from exc

    now = datetime.now(timezone.utc)
    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, sorted(names)[0])]))
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    cert_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    _write_key(key_path, leaf_key)
    return ca_cert_path, cert_path, key_path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Issue a trusted-on-your-LAN Fulloch HTTPS certificate")
    parser.add_argument("--cert-dir", type=Path, default=root / "data" / "certs")
    parser.add_argument("--host", action="append", default=[], help="extra DNS name to include (repeatable)")
    parser.add_argument("--ip", action="append", default=[], help="extra IP address to include (repeatable)")
    parser.add_argument("--force", action="store_true", help="replace the existing dashboard certificate")
    args = parser.parse_args()

    try:
        ca_cert, cert, key = issue_certificates(args.cert_dir, args.host, args.ip, args.force)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Created a CA-signed dashboard certificate:")
    print(f"  Dashboard certificate: {cert}")
    print(f"  Dashboard key:         {key}")
    print(f"  Public CA to install:  {ca_cert}")
    print()
    print("Install ONLY the public CA certificate on each client device:")
    print(f"  Linux:   sudo cp {ca_cert} /usr/local/share/ca-certificates/ && sudo update-ca-certificates")
    print(f"  macOS:   sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain {ca_cert}")
    print(f"  Windows: certutil -addstore -f Root {ca_cert}")
    print("  iOS/iPadOS: AirDrop/email the .crt, install its profile, then enable full trust in")
    print("              Settings > General > About > Certificate Trust Settings.")
    print("  Android: install the .crt as a CA certificate in Settings > Security.")
    print()
    print("Restart Fulloch after replacing the dashboard certificate. Never copy or install")
    print(f"the private CA key: {args.cert_dir / CA_KEY_FILENAME}")


if __name__ == "__main__":
    main()
