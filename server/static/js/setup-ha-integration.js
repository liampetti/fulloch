import { postJSON } from './setup-common.js';

// Shared wizard/settings card: token is only displayed after explicit creation.
export function integrationCard() {
  return `<div class="ha-integration-card">
    <div class="ha-integration-heading"><span aria-hidden="true">↔</span><div><strong>Connect from Home Assistant</strong><p>Optional: add Fulloch as an integration in Home Assistant.</p></div></div>
    <p>In Home Assistant, add the Fulloch integration, then enter Fulloch’s address, port and a Fulloch token below. This is different from the Home Assistant access token above.</p>
    <button type="button" id="ha-create-integration">Create Fulloch token</button>
    <div id="ha-integration-result" hidden>
      <p id="ha-integration-address"></p>
      <label for="ha-integration-token">Fulloch integration token</label>
      <div class="ha-token-row"><input id="ha-integration-token" type="text" readonly autocomplete="off" spellcheck="false"><button type="button" id="ha-copy-integration">Copy</button></div>
      <p class="help">Paste this token into Home Assistant. Keep it secret; copy it now—it will not be shown again after leaving this page. Existing tokens still work.</p>
      <p class="help">The integration API uses HTTP. Use it only on a trusted network.</p>
      <p id="ha-integration-network" class="help warning"></p>
    </div>
    <p id="ha-integration-status" class="help" role="status"></p>
  </div>`;
}

export function wireIntegrationCard() {
  const create = document.getElementById('ha-create-integration');
  const result = document.getElementById('ha-integration-result');
  const tokenField = document.getElementById('ha-integration-token');
  const status = document.getElementById('ha-integration-status');
  create.addEventListener('click', async () => {
    create.disabled = true; status.textContent = 'Creating token…';
    try {
      const response = await postJSON('/setup/integration-token');
      if (!response.ok) throw new Error('Could not create a token. Please try again.');
      const data = await response.json();
      if (!result.isConnected) return;
      tokenField.value = data.token;
      const host = location.hostname;
      document.getElementById('ha-integration-address').textContent =
        `Fulloch address: ${host} · Integration API port: ${data.port}`;
      const warnings = [];
      if (!data.enabled) warnings.push('The integration API is disabled in config.yml; enable it and restart Fulloch.');
      if (['127.0.0.1', 'localhost', '::1', '[::1]'].includes(data.bind_host))
        warnings.push('Fulloch is listening only on this machine. Set Dashboard Host to 0.0.0.0 and restart to allow Home Assistant on another device to connect.');
      if (['localhost', '127.0.0.1', '::1', '[::1]'].includes(host))
        warnings.push('If Home Assistant runs on another device, use Fulloch’s LAN IP address instead of localhost.');
      document.getElementById('ha-integration-network').textContent = warnings.join(' ');
      result.hidden = false;
      create.hidden = true;
      status.textContent = 'Token created. Home Assistant can connect once Fulloch has finished starting.';
    } catch (error) {
      if (status.isConnected) status.textContent = error.message || 'Could not create a token.';
    } finally { create.disabled = false; }
  });
  document.getElementById('ha-copy-integration').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(tokenField.value);
      status.textContent = 'Copied. Paste it into the Fulloch integration in Home Assistant.';
    } catch {
      tokenField.focus(); tokenField.select();
      status.textContent = 'Select and copy the token above.';
    }
  });
}
