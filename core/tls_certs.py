"""Self-signed HTTPS certificate for the dashboard, generated on first launch.

Browsers refuse microphone access (`getUserMedia`) on a plain-HTTP origin
unless it's `localhost`/`127.0.0.1` — so the browser satellite silently can't
get mic permission for anyone reaching the dashboard from a phone or another
machine on the LAN. There's no CA that can sign a certificate for a private
LAN IP, so this generates a self-signed one; every browser shows a one-time
"not private" warning on first visit per device — expected, not a bug.
"""

import ipaddress
import logging
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

CERT_FILENAME = "dashboard.crt"
KEY_FILENAME = "dashboard.key"
# No renewal flow exists yet, so this is long-lived by design (common practice
# for self-signed LAN certs — e.g. router/NAS admin UIs do the same). Browsers
# don't apply the public-CA lifetime caps to a manually-trusted self-signed cert.
_VALIDITY_DAYS = 3650


def _detect_local_ips() -> set:
    """Best-effort discovery of this host's LAN IPv4 addresses, no extra deps.

    The UDP "connect" never sends a packet (UDP is connectionless) — it just
    asks the OS which local address it would route through to reach the
    target, which is the LAN IP if one exists. Silently skipped if no network
    route is available; the cert still covers localhost/127.0.0.1.
    """
    ips = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        _, _, addrs = socket.gethostbyname_ex(socket.gethostname())
        ips.update(addrs)
    except OSError:
        pass
    return ips


def ensure_self_signed_cert(certs_dir: str = "./data/certs") -> Tuple[str, str]:
    """Create data/certs/dashboard.{crt,key} if they don't already exist.

    Returns (cert_path, key_path) either way, so the caller can always wire
    them into config.yml. Idempotent: an existing pair is left untouched, so a
    user who's dropped in their own cert (e.g. via mkcert) never gets it
    silently overwritten.
    """
    certs = Path(certs_dir)
    certs.mkdir(parents=True, exist_ok=True)
    cert_path = certs / CERT_FILENAME
    key_path = certs / KEY_FILENAME

    if cert_path.is_file() and key_path.is_file():
        return str(cert_path), str(key_path)

    return _generate(cert_path, key_path)


def regenerate_self_signed_cert(certs_dir: str = "./data/certs") -> Tuple[str, str]:
    """Force a fresh cert/key pair, overwriting any existing one.

    Unlike `ensure_self_signed_cert`, this is not idempotent — every browser
    that already trusted the old cert will see the "not private" warning
    again on next visit. Used by the dashboard's manual "regenerate
    certificate" action (e.g. after the LAN IP changes and old SANs go
    stale), not on normal startup.
    """
    certs = Path(certs_dir)
    certs.mkdir(parents=True, exist_ok=True)
    cert_path = certs / CERT_FILENAME
    key_path = certs / KEY_FILENAME
    return _generate(cert_path, key_path)


def _generate(cert_path: Path, key_path: Path) -> Tuple[str, str]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "fulloch.local")])

    sans: list = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        sans.append(x509.IPAddress(ipaddress.ip_address("::1")))
    except ValueError:
        pass
    for ip in sorted(_detect_local_ips()):
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            continue
    hostname = socket.gethostname()
    if hostname and hostname.lower() != "localhost":
        sans.append(x509.DNSName(hostname))

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=_VALIDITY_DAYS))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .sign(key, hashes.SHA256())
    )

    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    logger.info(
        "Generated a self-signed HTTPS certificate at %s (SANs: %s) — browsers "
        "will show a one-time security warning on first visit; that's expected "
        "for a private LAN certificate, not a bug.",
        cert_path,
        ", ".join(str(s.value) for s in sans),
    )
    return str(cert_path), str(key_path)
