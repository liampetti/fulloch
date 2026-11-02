"""Optional Obsidian bridge management, kept out of the normal setup UI.

Run with --help. Talks to an already-running dashboard; never starts models.
"""

import argparse
import getpass
import http.cookiejar
import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://localhost:8765", help="Fulloch dashboard URL")
    parser.add_argument("--login", action="store_true", help="Prompt for the dashboard password")
    parser.add_argument("--ca", help="Trusted private CA certificate (preferred for local HTTPS)")
    parser.add_argument("--insecure", action="store_true", help="Explicitly trust a self-signed local server")
    parser.add_argument("--vault-path", help="Vault directory as seen by the Fulloch server/container")
    parser.add_argument("--plugin-zip", type=Path, help="Save the plugin archive here (must not exist)")
    parser.add_argument("--show-token", action="store_true", help="Print the secret plugin token for manual pairing")
    args = parser.parse_args(argv)
    if not args.url.startswith(("http://", "https://")):
        parser.error("--url must be an HTTP(S) dashboard address")
    if args.plugin_zip and args.plugin_zip.exists():
        parser.error("--plugin-zip already exists; choose a new path")
    context = ssl.create_default_context(cafile=args.ca)
    if args.insecure:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
        urllib.request.HTTPSHandler(context=context),
    )

    def request(path, body=None, *, binary=False):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.url.rstrip("/") + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=30) as response:
            payload = response.read()
        return payload if binary else json.loads(payload)

    try:
        if args.login:
            request("/auth/login", {"password": getpass.getpass("Dashboard password: ")})
        if args.vault_path:
            request("/api/setup/obsidian-vault", {"path": args.vault_path})
        if args.plugin_zip:
            archive = request("/api/obsidian/plugin.zip", binary=True)
            with args.plugin_zip.open("xb") as file:
                file.write(archive)
            print(f"Saved plugin archive to {args.plugin_zip}")
        status = request("/api/obsidian/status")
        print("Bridge:", "connected" if status.get("connected") else "disconnected")
        print("Vault:", status.get("vault_path") or "not configured")
        if args.show_token:
            print("Plugin token (keep secret):", request("/api/obsidian/show-token", {}).get("token", ""))
    except (OSError, ValueError, urllib.error.URLError) as exc:
        parser.exit(1, f"Obsidian setup failed: {exc}\nUse --login if the dashboard requires a password.\n")


if __name__ == "__main__":
    main()
