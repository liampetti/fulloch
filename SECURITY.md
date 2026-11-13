# Security Policy

## Design Philosophy

Fulloch is designed with privacy as a core principle. The local stack processes speech, language, and stored data on your device:

- **Speech Recognition**: Qwen3 ASR runs entirely on-device
- **Text-to-Speech**: Qwen3 TTS Base runs entirely on-device (voice cloned from a local `data/voices/<name>.{wav,txt}` reference pair; no audio leaves your machine)
- **Language Model**: Local GPU models run on-device via llama.cpp; Lite uses local Laya and regex routing.
- **Optional remote services**: Remote LLM, Spotify, web search, academic research, travel and finance integrations contact their providers when enabled or configured.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

Contact the maintainers through GitHub or raise an issue on the project. For a
sensitive vulnerability, use GitHub's private reporting or a private contact
method if available; do not include exploitable details or secrets in a public
issue. Include reproduction steps where possible and allow time for a fix.

## Security Considerations

### Configuration and User Data

- `data/config.yml` contains configuration and may contain satellite tokens — treat it as sensitive
- `data/credentials.json` contains credentials such as API keys and password hashes — never commit it
- `data/notes/` holds user-volunteered notes — including `fulloch_facts.md` (long-term personal facts auto-injected into the chat prompt) and any daily journal entries
- All of the above are excluded from git via `.gitignore`

### Network Services

Fulloch connects to a small number of external services:

| Service | Connection Type | Data Sent |
|---------|----------------|-----------|
| Home Assistant | Local HTTP (REST) | Entity commands, calendar queries, weather forecasts |
| SearXNG | Local HTTP | Search queries (only when `search:` is configured) |
| Remote LLM | Configured OpenAI-compatible endpoint | Conversation prompts and tool observations |
| Spotify | Spotify Web API | Search and playback requests, when `spotify:` is configured |
| Academic research | Semantic Scholar, arXiv, OpenAlex | Paper search queries, only with `research.enabled: true` |
| Travel and finance | SerpApi | Travel or market queries when a SerpApi key is configured |
| Wolfram | Wolfram Alpha | Queries when a Wolfram app ID is configured |
| MCP | User-configured servers | Tool requests when an MCP server is configured |

Home Assistant is the sole smart-home control backend. Spotify search uses Spotify's Web API directly, then hands resolved playback to Home Assistant when a target is available.

### Best Practices

1. **Network Isolation**: Run Fulloch on a trusted local network
2. **Credential Rotation**: Regularly rotate API keys and tokens
3. **Minimal Permissions**: Use read-only API access where possible
4. **Update Dependencies**: Keep dependencies updated for security patches
5. **Restrict sensitive entities from voice**: In the dashboard's **Entities** tab, switch off any entity that should not be voice-controllable (door locks, alarms). Fulloch then refuses to control it by voice while it stays usable in the dashboard. Changes apply immediately and persist in `data/voice_denylist.json`. This is separate from HA's "Expose to Assist" setting, which Fulloch cannot read.

### Credentials

Fulloch stores secrets in `data/credentials.json` (written by the setup wizard, never committed). Treat it as sensitive:
- **Home Assistant** long-lived token (`ha_token`).
- **Dashboard password** (`dashboard_password`) — PBKDF2-SHA256 hash. Strongly recommended before binding to a non-loopback address, but not enforced; without one the dashboard is unauthenticated. Set it in Settings → Dashboard.
- **Remote LLM API key** (`llm_api_key`) if using an OpenAI-compatible endpoint.
- **Obsidian plugin token** (`obsidian_token`) if using the Obsidian bridge.
- **Home Assistant integration tokens** (`integration_tokens`) — bearer tokens for the separate plain-HTTP integration API. Keep its bind address on a trusted network and rotate tokens if exposed.

> During initial first-run setup no password is set yet, so keep the dashboard on a trusted/loopback network until the wizard completes.

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |
| Older   | No        |

Only the latest version receives security updates.
