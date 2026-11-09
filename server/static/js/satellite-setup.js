// Waveshare provisioning; the server owns credentials and transaction state.
const BASE = '/api/satellite-provisioning';
const HEADERS = { 'X-Fulloch-Provisioning': '1' };
const RECOVERY = 'USB writes cannot be undone. Re-enter your secrets and flash again if the device does not connect.';
const ESP32_USB = { usbVendorId: 0x303a, usbProductId: 0x1001 };

export function createSatelliteSetup({ root = document, serial = navigator.serial,
  secure = window.isSecureContext, request = window.fetch.bind(window),
  loadFlasher = () => import('../vendor/esptool-js/bundle.js'), onSuccess = () => {} } = {}) {
  const dialog = root.getElementById('sat-setup');
  const form = root.getElementById('sat-setup-form');
  const fields = root.getElementById('sat-setup-fields');
  const prepare = root.getElementById('sat-setup-prepare');
  const cancel = root.getElementById('sat-setup-cancel');
  const support = root.getElementById('sat-setup-support');
  const message = root.getElementById('sat-setup-message');
  const progress = root.getElementById('sat-setup-progress');
  const input = name => form.elements.namedItem(name);
  let transaction = null, artifact = null, manifest = null, transport = null;
  let cancelled = false, busy = false, finished = false, inflight = false;

  function status(text) { message.textContent = text; }
  function resetSecrets() {
    for (const name of ['wifi_password', 'spotify_secret', 'spotify_token']) input(name).value = '';
  }
  function current() { if (cancelled) throw new Error('cancelled'); }
  async function api(path, options = {}) {
    const response = await request(BASE + path, {
      cache: 'no-store', ...options, headers: { ...HEADERS, ...options.headers },
    });
    if (!response.ok) {
      let detail = 'The provisioning request failed. Check your password session and try again.';
      try { detail = (await response.json()).detail || detail; } catch (_) { /* no details */ }
      throw new Error(typeof detail === 'string' ? detail : 'Invalid provisioning request.');
    }
    return response;
  }
  async function cleanup() {
    const id = transaction;
    transaction = null;
    artifact = null;
    manifest = null;
    if (id) {
      try { await api(`/transactions/${encodeURIComponent(id)}`, { method: 'DELETE', keepalive: true }); }
      catch (_) { /* May already have expired or committed. */ }
    }
  }
  function idle() {
    busy = false;
    fields.disabled = false;
    prepare.disabled = !secure || !serial;
    prepare.hidden = false;
    cancel.textContent = finished || cancelled ? 'Close' : 'Cancel';
  }
  function payload() {
    const url = new URL(input('server_url').value);
    if (url.protocol !== 'wss:' || !url.hostname || url.username || url.password || url.hash || url.search || url.pathname !== '/ws/satellite-v2') {
      throw new Error('Use a reachable wss:// address ending in /ws/satellite-v2, without credentials or a query.');
    }
    const player = input('music').value;
    const config = {
      schema_version: 1,
      device: { id: '', name: input('device_name').value.trim(), timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || '' },
      server: { url: url.href, certificate_pem: '' },
      audio: { speaker_volume_percent: Number(input('volume').value), aec_enabled: input('aec').checked },
      display: { rotation_degrees: Number(input('rotation').value) },
      music: { player },
    };
    const credentials = { schema_version: 1, wifi: { ssid: input('wifi_ssid').value, password: input('wifi_password').value } };
    if (player === 'sonos') {
      if (!input('sonos_host').value.trim()) throw new Error('Enter the Sonos host.');
      config.sonos = { host: input('sonos_host').value.trim() };
    }
    if (player === 'spotify') {
      if (!input('spotify_id').value.trim() || !input('spotify_secret').value || !input('spotify_token').value) throw new Error('Enter all Spotify credentials.');
      credentials.spotify = { client_id: input('spotify_id').value.trim(), client_secret: input('spotify_secret').value, refresh_token: input('spotify_token').value };
    }
    return { config, credentials, auto_setup: true };
  }
  function musicChanged() {
    root.getElementById('sat-setup-sonos').hidden = input('music').value !== 'sonos';
    root.getElementById('sat-setup-spotify').hidden = input('music').value !== 'spotify';
    if (input('music').value === 'sonos') discoverSonos();
  }
  function options(id, values) {
    const list = root.getElementById(id);
    list.replaceChildren(...values.map(value => {
      const option = root.createElement('option'); option.value = value; return option;
    }));
  }
  async function discoverWifi() {
    const note = root.getElementById('sat-setup-wifi-status');
    note.textContent = 'Looking for Wi-Fi networks…';
    try {
      const { networks } = await (await api('/waveshare/wifi')).json();
      options('sat-setup-networks', networks);
      note.textContent = networks.length ? 'Choose a network or type its name.' : 'No networks detected. Type the Wi-Fi network name.';
    } catch (_) { note.textContent = 'Wi-Fi discovery unavailable. Type the network name.'; }
  }
  async function discoverSonos() {
    const note = root.getElementById('sat-setup-sonos-status');
    note.textContent = 'Looking for Sonos speakers…';
    try {
      const { hosts } = await (await api('/waveshare/sonos')).json();
      options('sat-setup-sonos-hosts', hosts);
      if (hosts.length && !input('sonos_host').value) input('sonos_host').value = hosts[0];
      note.textContent = hosts.length ? 'Choose a detected speaker or enter another host.' : 'No Sonos speakers detected. Enter the host manually.';
    } catch (_) { note.textContent = 'Sonos discovery unavailable. Enter the host manually.'; }
  }
  input('music').addEventListener('change', musicChanged);
  async function open() {
    if (busy) return;
    cancelled = false; finished = false;
    form.reset(); musicChanged(); resetSecrets();
    progress.hidden = true; status('');
    if (window.location.hostname) input('server_url').value = `wss://${window.location.host}/ws/satellite-v2`;
    idle();
    prepare.disabled = true;
    support.textContent = !secure
      ? 'USB setup needs HTTPS in a secure context. Open the dashboard using its trusted HTTPS certificate, or use the CLI fallback below.'
      : !serial ? 'WebSerial is unavailable. Use a supported Chromium browser over trusted HTTPS, or use the CLI fallback below.' : 'The dashboard certificate or its local CA will be used automatically.';
    dialog.showModal();
    try {
      const response = await request('/status', { cache: 'no-store' });
      const state = await response.json();
      if (!response.ok || !state.auth_enabled) {
        prepare.disabled = true;
        support.textContent = 'Enable a dashboard password in Settings → Security, then sign in before setup.';
      } else {
        prepare.disabled = !secure || !serial;
        discoverWifi();
      }
    } catch (_) {
      prepare.disabled = true;
      support.textContent = 'Cannot verify dashboard password protection. Check the connection and sign in.';
    }
  }
  async function downloadImages() {
    const images = [];
    for (const image of manifest.images) {
      current();
      const bytes = new Uint8Array(await (await api(`/waveshare/images/${encodeURIComponent(image.file)}`)).arrayBuffer());
      const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
      if (bytes.length !== image.size || hash !== image.sha256) throw new Error('Firmware integrity check failed.');
      images.push({ data: bytes, address: image.offset });
    }
    images.push({ data: artifact, address: manifest.nvs.offset });
    return images;
  }
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (busy || !secure || !serial || !form.reportValidity()) return;
    let data;
    try { data = payload(); } catch (error) { status(error.message); return; }
    const showAll = input('show_all_ports').checked;
    busy = true; inflight = true; cancelled = false; fields.disabled = true; prepare.disabled = true;
    let connecting = false;
    try {
      // WebSerial requires the chooser directly in the user's gesture.
      const port = await serial.requestPort(showAll ? {} : { filters: [ESP32_USB] });
      current();
      progress.value = 0; progress.hidden = false;
      status('Enabling authentication and validating the device bundle…');
      await api('/migration', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ confirm_tokenless_reprovisioning: true }) });
      current();
      const response = await api('/waveshare/transactions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
      transaction = (await response.json()).transaction_id;
      data = null; resetSecrets();
      current();
      manifest = await (await api('/waveshare/manifest')).json();
      current();
      artifact = new Uint8Array(await (await api(`/transactions/${encodeURIComponent(transaction)}/nvs`)).arrayBuffer());
      current();
      if (artifact.length !== manifest.nvs.size) throw new Error('The NVS image does not match the firmware manifest.');
      status('Checking firmware and connecting to the board…');
      const state = await (await api(`/transactions/${encodeURIComponent(transaction)}`)).json();
      current();
      if (state.state !== 'ready') throw new Error('The setup transaction expired. Start again.');
      const images = await downloadImages();
      current();
      const { ESPLoader, Transport } = await loadFlasher();
      current();
      transport = new Transport(port, false);
      // Never allow serial traces to print NVS bytes or credentials.
      transport.trace = () => {};
      const loader = new ESPLoader({ transport, baudrate: 115200, debugLogging: false,
        terminal: { clean() {}, write() {}, writeLine() {} } });
      connecting = true;
      await loader.main();
      connecting = false;
      current();
      if (loader.chip?.CHIP_NAME !== 'ESP32-S3') throw new Error('This is not an ESP32-S3. Use the Waveshare Rev 2 board.');
      progress.hidden = false;
      await loader.writeFlash({ fileArray: images, flashSize: 'keep', flashMode: 'keep', flashFreq: 'keep', eraseAll: false, compress: true,
        reportProgress(index, written, total) {
          current();
          const completed = images.slice(0, index).reduce((sum, image) => sum + image.data.length, 0);
          const size = images.reduce((sum, image) => sum + image.data.length, 0);
          progress.value = Math.min(99, Math.round((completed + written / total * images[index].data.length) / size * 100));
          status(`Flashing image ${index + 1} of ${images.length}: ${progress.value}%`);
        } });
      current();
      // esptool-js 0.7.1's hard_reset only releases RTS, so explicitly pulse
      // EN with GPIO0 released. This starts the flashed app, not the ROM loader.
      let resetFailed = false;
      try {
        await transport.setDTR(false);
        try {
          await transport.setRTS(true);
          await new Promise(resolve => setTimeout(resolve, 100));
        } finally {
          await transport.setRTS(false);
        }
      } catch (_) { resetFailed = true; }
      // A successful reset may disconnect the USB serial device itself.
      try { await transport.disconnect(); } catch (_) { /* Device may have rebooted. */ }
      transport = null;
      artifact = null;
      current();
      status(resetFailed
        ? 'Flash complete, but automatic restart failed. Press the device reset button; waiting for authentication…'
        : 'Flash complete. Waiting for the expected device to authenticate…');
      while (true) {
        current();
        const result = await (await api(`/transactions/${encodeURIComponent(transaction)}`)).json();
        current();
        if (result.state === 'confirmed') break;
        if (result.state !== 'ready') throw new Error(`The device did not connect before the setup timeout. ${result.recovery || ''}`);
        await new Promise(resolve => setTimeout(resolve, 1500));
      }
      transaction = null; manifest = null; finished = true; progress.value = 100;
      status('Setup complete: the expected device authenticated successfully.');
      idle(); prepare.hidden = true; onSuccess();
    } catch (error) {
      if (transport) { try { await transport.disconnect(); } catch (_) {} transport = null; }
      await cleanup();
      if (!cancelled) {
        if (error.name === 'NotFoundError') {
          status('No USB device selected. Re-enter your secrets and try again.');
        } else if (connecting && (error.name === 'NetworkError' || /failed to open serial port/i.test(error.message))) {
          const platform = navigator.userAgentData?.platform || navigator.platform || '';
          const linux = /linux/i.test(platform)
            ? ' On Linux, check serial-port permissions: run sudo usermod -aG dialout "$USER", then log out completely and back in before reopening the browser.'
            : '';
          status(`Could not open or connect to the USB serial port. No firmware was written. Close serial monitors and other apps using the board, reconnect its USB cable, and try again.${linux} If it still fails, check that another service is not using the port. Re-enter your secrets to retry.`);
        } else {
          status(`${error.message} ${RECOVERY}`);
        }
      }
      idle();
    } finally {
      inflight = false; data = null; resetSecrets();
    }
  });
  async function stop() {
    if (finished || cancelled || (!busy && !inflight)) {
      if (dialog.open) dialog.close();
      return;
    }
    cancelled = true; resetSecrets();
    status(`Setup cancelled. ${RECOVERY}`);
    cancel.textContent = 'Close';
    await cleanup();
    if (transport) { try { await transport.disconnect(); } catch (_) {} transport = null; }
    if (!inflight) idle();
  }
  cancel.addEventListener('click', stop);
  dialog.addEventListener('cancel', event => { event.preventDefault(); stop(); });
  window.addEventListener('pagehide', () => { cancelled = true; cleanup(); resetSecrets(); });
  return { open };
}
