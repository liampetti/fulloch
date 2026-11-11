"""The test suite must not depend on external network services."""

import socket

import pytest


def test_external_dns_is_blocked():
    with pytest.raises(OSError, match="External network access is disabled"):
        socket.getaddrinfo("huggingface.co", 443)


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_external_ip_connections_are_blocked(method):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(OSError, match="External network access is disabled"):
            getattr(sock, method)(("192.0.2.1", 443))


def test_loopback_resolution_is_allowed():
    assert socket.getaddrinfo("localhost", 80)
