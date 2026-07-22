# Security Policy

## Design Philosophy

Fulloch is designed with privacy as a core principle. All processing happens locally on your device:

- **Speech Recognition**: Qwen3 ASR runs entirely on-device
- **Text-to-Speech**: Qwen3 TTS Base runs entirely on-device (voice cloned from a local `data/voices/<name>.{wav,txt}` reference pair; no audio leaves your machine)
- **Language Model**: Qwen3.5 9B runs entirely on-device via llama.cpp
- **No Cloud Dependencies**: No data is sent to external servers for AI processing

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email the maintainers directly with details of the vulnerability
3. Include steps to reproduce if possible
4. Allow reasonable time for a fix before public disclosure

## Security Considerations

### Configuration and User Data

- `data/config.yml` contains configuration choices (no secrets) — safe but still excluded from git
- `data/credentials.json` contains all secrets (tokens, passwords) — never committed
- `data/notes/` holds user-volunteered notes — including `fulloch_facts.md` (long-term personal facts auto-injected into the chat prompt) and any daily journal entries
- All of the above are excluded from git via `.gitignore`

### Network Services

Fulloch connects to a small number of external services:

| Service | Connection Type | Data Sent |
|---------|----------------|-----------|
| Home Assistant | Local HTTP (REST) | Entity commands, calendar queries, weather forecasts |
| SearXNG | Local HTTP | Search queries (only when `search:` is configured) |

Home Assistant is the sole smart-home backend; all third-party device protocols (Spotify, Hue, Calendar, etc.) terminate inside HA, not Fulloch.

### Best Practices

1. **Network Isolation**: Run Fulloch on a trusted local network
2. **Credential Rotation**: Regularly rotate API keys and tokens
3. **Minimal Permissions**: Use read-only API access where possible
4. **Update Dependencies**: Keep dependencies updated for security patches
5. **Restrict sensitive entities from voice**: In the dashboard's **Entities** tab, switch off any entity that should not be voice-controllable (door locks, alarms). Fulloch then refuses to control it by voice while it stays usable in the dashboard. Changes apply immediately and persist in `data/voice_denylist.json`. This is separate from HA's "Expose to Assist" setting, which Fulloch cannot read.

### Credentials

Fulloch stores secrets in `data/credentials.json` (written by the setup wizard, never committed). Treat it as sensitive:
- **Home Assistant** long-lived token (`ha_token`).
- **Dashboard password** (`dashboard_password`) — PBKDF2-SHA256 hash. Required when binding to a non-loopback address; set via the wizard's finish step.
- **Remote LLM API key** (`llm_api_key`) if using an OpenAI-compatible endpoint.
- **Obsidian plugin token** (`obsidian_token`) if using the Obsidian bridge.

> During initial first-run setup no password is set yet, so keep the dashboard on a trusted/loopback network until the wizard completes.

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |
| Older   | No        |

Only the latest version receives security updates.
