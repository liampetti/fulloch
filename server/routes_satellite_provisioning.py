"""Password-protected satellite provisioning resources."""

from copy import deepcopy

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from . import (
    satellite_authorization,
    satellite_bundle,
    satellite_setup_discovery,
    satellite_tokens,
    satellite_usb,
    satellite_waveshare,
)

_NO_CACHE = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


def register_satellite_provisioning_routes(app: FastAPI, context) -> None:
    """Serve only checked, public Rev 2 release artifacts to authorized clients."""

    def release():
        try:
            return satellite_usb.load_release()
        except satellite_usb.ArtifactError as error:
            raise HTTPException(
                status_code=503, detail="Rev 2 firmware release is unavailable or invalid"
            ) from error

    @app.get("/api/satellite-provisioning/waveshare/manifest")
    def firmware_manifest(request: Request):
        satellite_authorization.authorize(request, context)
        manifest, _ = release()
        return JSONResponse(manifest, headers=_NO_CACHE)

    @app.get("/api/satellite-provisioning/waveshare/wifi")
    def wifi(request: Request):
        satellite_authorization.authorize(request, context)
        return JSONResponse({"networks": satellite_setup_discovery.wifi_networks()}, headers=_NO_CACHE)

    @app.get("/api/satellite-provisioning/waveshare/sonos")
    def sonos(request: Request):
        satellite_authorization.authorize(request, context)
        return JSONResponse({"hosts": satellite_setup_discovery.sonos_hosts()}, headers=_NO_CACHE)

    @app.get("/api/satellite-provisioning/waveshare/images/{name}")
    def firmware_image(name: str, request: Request):
        satellite_authorization.authorize(request, context)
        _, images = release()
        if name not in images:
            raise HTTPException(status_code=404, detail="Firmware image not found")
        return Response(images[name], media_type="application/octet-stream", headers=_NO_CACHE)

    async def body(request):
        satellite_authorization.authorize(request, context)
        try:
            data = await request.json()
        except ValueError:
            raise HTTPException(status_code=400, detail="Expected a JSON object") from None
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Expected a JSON object")
        return data

    @app.post("/api/satellite-provisioning/migration")
    async def migrate(request: Request):
        data = await body(request)
        if (
            set(data) != {"confirm_tokenless_reprovisioning"}
            or data.get("confirm_tokenless_reprovisioning") is not True
        ):
            raise HTTPException(
                status_code=400,
                detail="Confirm that all existing tokenless devices must be reprovisioned",
            )
        # Preserve connected tokenless identities before enforcement disconnects them.
        sessions = getattr(context.assistant, "satellites", {}) or {}
        known = getattr(app.state, "satellite_setup_known_ids", set())
        known.update(session.device_id for session in sessions.values() if getattr(session, "device_id", None))
        app.state.satellite_setup_known_ids = known
        satellite_tokens.enable_enforcement(context.config_path, confirmed=True)
        await context.revalidate_satellite_connections()
        return JSONResponse({"token_required": True}, headers=_NO_CACHE)

    @app.delete("/api/satellite-provisioning/devices/{device_id:path}/token")
    async def revoke(device_id: str, request: Request):
        satellite_authorization.authorize(request, context)
        removed = context.satellite_provisioning.revoke_device(device_id)
        await context.revalidate_satellite_connections()
        return JSONResponse({"revoked": removed}, headers=_NO_CACHE)

    @app.post("/api/satellite-provisioning/waveshare/transactions")
    async def prepare(request: Request):
        data = await body(request)
        if set(data) not in ({"config", "credentials"}, {"config", "credentials", "auto_setup"}) or not all(
            isinstance(data[key], dict) for key in ("config", "credentials")
        ) or ("auto_setup" in data and data["auto_setup"] is not True):
            raise HTTPException(status_code=400, detail="Expected config and credentials objects")
        config, credentials = deepcopy(data["config"]), deepcopy(data["credentials"])
        try:
            server = config["server"]
            if not isinstance(server, dict):
                raise ValueError("server must be an object")
            identity_certificate = None
            if not server.get("certificate_pem"):
                certificate = getattr(app.state, "satellite_dashboard_certificate", None)
                if not certificate:
                    raise ValueError("Supply the public dashboard certificate")
                server["certificate_pem"] = satellite_waveshare.dashboard_trust_certificate(
                    certificate, getattr(app.state, "satellite_dashboard_ca_path", None)
                )
                if server["certificate_pem"] != certificate:
                    identity_certificate = certificate
            auto_setup = data.get("auto_setup") is True
            if auto_setup:
                config["device"]["timezone"] = satellite_setup_discovery.posix_timezone(config["device"]["timezone"])
                config["device"]["id"] = "validation-placeholder"
            device_id = config["device"]["id"]
            if not isinstance(device_id, str) or device_id != device_id.strip():
                raise ValueError("Device ID must not have surrounding whitespace")
            if "fulloch" in credentials:
                raise ValueError("Device tokens are generated by the server")
            credentials["fulloch"] = {"device_token": "validation-placeholder"}
            satellite_waveshare.validate_bundle(
                config, credentials, identity_certificate=identity_certificate
            )
            release()

            def generate(token):
                credentials["fulloch"]["device_token"] = token
                if identity_certificate:
                    return satellite_waveshare.generate_nvs(
                        config, credentials, identity_certificate=identity_certificate
                    )
                return satellite_waveshare.generate_nvs(config, credentials)

            if auto_setup:
                sessions = getattr(context.assistant, "satellites", {}) or {}
                occupied = [getattr(session, "device_id", None) for session in sessions.values()]
                occupied.extend(getattr(app.state, "satellite_setup_known_ids", set()))

                def generate_new(new_id, token):
                    config["device"]["id"] = new_id
                    return generate(token)

                transaction = context.satellite_provisioning.prepare_new(config["device"]["name"], generate_new, occupied=occupied)
            else:
                transaction = context.satellite_provisioning.prepare(device_id, generate)
        except (KeyError, TypeError, OSError, satellite_bundle.BundleError, ValueError) as error:
            # Validation messages contain field names, not supplied values. Never
            # reflect arbitrary input or filesystem exception details.
            if isinstance(error, satellite_bundle.BundleError) and str(error) in {
                "server certificate does not match the selected server host",
                "server certificate is expired or not yet valid",
                "server certificate needs an IP address subject alternative name",
                "Cannot verify the dashboard certificate with its local CA; check the public CA certificate",
            }:
                detail = str(error)
            elif isinstance(error, ValueError) and str(error) in {
                "server must be an object",
                "Supply the public dashboard certificate",
                "Device ID must not have surrounding whitespace",
                "Device tokens are generated by the server",
                "mandatory token migration must be enabled first",
                "a provisioning attempt is already active for this device",
                "Unable to determine browser timezone",
                satellite_setup_discovery.UNSUPPORTED_TIMEZONE,
            }:
                detail = str(error)
            else:
                detail = "Invalid provisioning data or unavailable certificate"
            raise HTTPException(status_code=400, detail=detail) from None
        return JSONResponse({"transaction_id": transaction, "state": "ready"}, headers=_NO_CACHE)

    @app.get("/api/satellite-provisioning/transactions/{transaction_id}")
    def status(transaction_id: str, request: Request):
        satellite_authorization.authorize(request, context)
        state = context.satellite_provisioning.status(transaction_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Transaction not found")
        recovery = None
        if state in ("expired", "cancelled"):
            recovery = "Re-enter secrets and flash again; USB writes cannot be rolled back."
        return JSONResponse({"state": state, "recovery": recovery}, headers=_NO_CACHE)

    @app.get("/api/satellite-provisioning/transactions/{transaction_id}/nvs")
    def nvs(transaction_id: str, request: Request):
        satellite_authorization.authorize(request, context)
        artifact = context.satellite_provisioning.artifact(transaction_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Provisioning artifact unavailable")
        return Response(artifact, media_type="application/octet-stream", headers=_NO_CACHE)

    @app.delete("/api/satellite-provisioning/transactions/{transaction_id}")
    def cancel(transaction_id: str, request: Request):
        satellite_authorization.authorize(request, context)
        if not context.satellite_provisioning.cancel(transaction_id):
            raise HTTPException(status_code=404, detail="Active transaction not found")
        return JSONResponse({"state": "cancelled"}, headers=_NO_CACHE)
