# Optional Obsidian bridge

Normal notes need only **Settings → General → Notes directory**. Mount that
directory into the container if necessary. It need not contain `.obsidian`.

The optional plugin provides active-note/selection context and editor actions.
Advanced users can configure it with the helper, without dashboard setup cards:

```bash
.venv/bin/python scripts/setup_obsidian.py --help
.venv/bin/python scripts/setup_obsidian.py \
  --url https://localhost:8765 --login --ca data/certs/fulloch-home-ca.crt \
  --vault-path /vault --plugin-zip ./fulloch-plugin.zip --show-token
```

Use the vault path **inside the server/container**. Extract the ZIP into
`<vault>/.obsidian/plugins/fulloch/`, enable the community plugin, and enter the
dashboard URL and printed secret token. Do not share that token. The helper
does not overwrite an existing output archive. Omit `--login` if no password
is configured. For a deliberately trusted local self-signed server, use
`--insecure` instead of `--ca`; this disables TLS certificate verification.

For differing host/container vault paths, set `obsidian.path_translation` in
`data/config.yml`. Other advanced bridge options remain documented in
`data/config.example.yml`. Existing bridge endpoints/configuration are retained
for compatibility. See `scripts/create_local_ca.py` for trusted LAN HTTPS.
