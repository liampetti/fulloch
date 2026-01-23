# Security Policy

## Design Philosophy

Fulloch is designed with privacy as a core principle. All processing happens locally on your device:

- **Speech Recognition**: Moonshine ASR runs entirely on-device
- **Text-to-Speech**: Kokoro TTS runs entirely on-device
- **Language Model**: Qwen runs entirely on-device via llama.cpp
- **No Cloud Dependencies**: No data is sent to external servers for AI processing

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email the maintainers directly with details of the vulnerability
3. Include steps to reproduce if possible
4. Allow reasonable time for a fix before public disclosure

## Security Considerations

### Configuration Files

- `data/config.yml` contains service credentials and should never be committed
- `.env` files contain sensitive environment variables
- Both files are excluded from git via `.gitignore`

### Network Services

Fulloch connects to external services for smart home control:

| Service | Connection Type | Data Sent |
|---------|----------------|-----------|
| Spotify | HTTPS API | Playback commands |
| Philips Hue | Local HTTP | Light commands |
| Google Calendar | HTTPS API | Calendar queries |
| SearXNG | Local HTTP | Search queries |
| LG ThinQ | HTTPS API | Appliance queries |
| WebOS TV | Local WebSocket | TV commands |
| Pioneer AVR | Local TCP | Audio commands |
| Airtouch | Local Discovery | HVAC commands |

### Best Practices

1. **Network Isolation**: Run Fulloch on a trusted local network
2. **Credential Rotation**: Regularly rotate API keys and tokens
3. **Minimal Permissions**: Use read-only API access where possible
4. **Update Dependencies**: Keep dependencies updated for security patches

### OAuth Tokens

- Google Calendar tokens are stored in `data/token.json`
- Spotify tokens are managed by the spotipy library
- Tokens should be treated as secrets and not shared

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |
| Older   | No        |

Only the latest version receives security updates.
