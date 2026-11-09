"""Browser-level coverage of the USB wizard, with mocked serial and HTTP boundaries."""

from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[1] / "server" / "static"


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as runtime:
        try:
            instance = runtime.chromium.launch()
        except Exception:
            pytest.skip("Playwright Chromium is not installed")
        yield instance
        instance.close()


@pytest.fixture
def page(browser):
    page = browser.new_page()
    html = (ROOT / "index.html").read_text()
    dialog = html[html.index('<dialog id="sat-setup"') : html.index("</dialog>") + 9]

    def handle(route):
        if route.request.url.endswith("/satellite-setup.js"):
            route.fulfill(
                status=200,
                content_type="text/javascript",
                body=(ROOT / "js/satellite-setup.js").read_text(),
            )
        else:
            route.fulfill(status=200, content_type="text/html", body=dialog)

    page.route("**/*", handle)
    page.goto("http://localhost/")
    yield page
    page.close()


HARNESS = """
async ({secure=true, supported=true, auth=true, fail=false, expire=false, expireAfter=false, pauseFlash=false, pausePrepare=false, prepareFail=false, openFail=false, selectCancel=false, wrongChip=false, resetFail=false, pauseConfirm=false} = {}) => {
  window.calls = []; window.writes = []; window.resetSignals = []; window.flashCount = 0;
  const bytes = new Uint8Array([1,2,3]);
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
    b => b.toString(16).padStart(2, '0')).join('');
  const manifest = {images: [{file:'application.bin', offset:131072, size:3, sha256:hash}], nvs:{size:3,offset:36864}};
  let checks = 0;
  const request = async (url, options={}) => {
    calls.push({url, method:options.method || 'GET', body:options.body});
    if (url === '/status') return Response.json({auth_enabled:auth});
    if (url.endsWith('/waveshare/wifi')) return Response.json({networks:['Home Wi-Fi']});
    if (url.endsWith('/waveshare/sonos')) return Response.json({hosts:['192.168.1.10']});
    if (url.endsWith('/migration')) return Response.json({token_required:true});
    if (url.endsWith('/waveshare/transactions')) return prepareFail ? Response.json({detail:'Invalid bundle'}, {status:400}) : Response.json({transaction_id:'abc',state:'ready'});
    if (url.endsWith('/waveshare/manifest')) {
      if (pausePrepare) await new Promise(resolve => { window.releasePrepare = resolve; });
      return Response.json(manifest);
    }
    if (url.endsWith('/nvs') || url.endsWith('/application.bin')) return new Response(bytes);
    if (options.method === 'DELETE') return Response.json({state:'cancelled'});
    if (url.endsWith('/transactions/abc')) {
      if (pauseConfirm && checks === 1) await new Promise(resolve => { window.releaseConfirm = resolve; });
      return Response.json({state:expire ? 'expired' : ++checks > 1 ? (expireAfter ? 'expired' : 'confirmed') : 'ready'});
    }
    throw Error('unexpected request');
  };
  class Transport {
    async setDTR(value) { resetSignals.push(['DTR', value]); }
    async setRTS(value) { resetSignals.push(['RTS', value]); if (resetFail && value) throw Error('reset failed'); }
    async disconnect() { resetSignals.push(['disconnect']); }
  }
  class ESPLoader {
    constructor() { this.chip = {CHIP_NAME:wrongChip ? 'ESP32' : 'ESP32-S3'}; }
    async main() { if (openFail) throw new DOMException("Failed to open serial port.", "NetworkError"); }
    async writeFlash(options) {
      flashCount++;
      writes.push(...options.fileArray.map(item => item.address));
      if (pauseFlash) await new Promise(resolve => { window.releaseFlash = resolve; });
      options.reportProgress(0,3,3);
      if (fail) throw Error('USB write failed');
    }
    async after() {}
  }
  const {createSatelliteSetup} = await import('/satellite-setup.js');
  window.wizard = createSatelliteSetup({secure, serial:supported ? {requestPort: async (options) => {
      window.portOptions = options;
      calls.push({url:'serial-chooser'});
      if (selectCancel) throw new DOMException('No port selected', 'NotFoundError');
      return {};
    }} : null,
    request, loadFlasher: async () => ({Transport, ESPLoader})});
  await wizard.open();
}
"""


def fill(page):
    page.locator('[name="device_name"]').fill("Kitchen")
    page.locator('[name="wifi_ssid"]').fill("network")
    page.locator('[name="wifi_password"]').fill("private-password")
    page.locator('[name="migration"]').check()


def test_success_requires_authenticated_confirmation_and_uses_manifest_offsets(page):
    page.evaluate(HARNESS, {})
    fill(page)
    page.locator("#sat-setup-prepare").click()
    page.get_by_text("Setup complete: the expected device authenticated successfully.").wait_for()
    assert page.evaluate("writes") == [131072, 36864]
    assert page.evaluate("resetSignals") == [["DTR", False], ["RTS", True], ["RTS", False], ["disconnect"]]
    assert page.evaluate("portOptions") == {"filters": [{"usbVendorId": 0x303A, "usbProductId": 0x1001}]}
    assert page.locator('[name="wifi_password"]').input_value() == ""
    assert page.evaluate("calls.filter(x => x.url.endsWith('/transactions/abc')).length") == 2
    assert page.evaluate("calls.findIndex(x => x.url === 'serial-chooser') < calls.findIndex(x => x.url.endsWith('/migration'))")
    assert page.locator('#sat-setup-progress').get_attribute('value') == '100'
    payload = page.evaluate("JSON.parse(calls.find(x => x.url.endsWith('/waveshare/transactions')).body)")
    assert payload["auto_setup"] is True
    assert payload["config"]["device"]["name"] == "Kitchen"
    assert payload["config"]["device"]["timezone"]
    assert payload["config"]["audio"]["speaker_volume_percent"] == 80
    assert payload["config"]["audio"]["aec_enabled"] is True


def test_reset_failure_requests_manual_reset_and_keeps_waiting(page):
    page.evaluate(HARNESS, {"resetFail": True, "pauseConfirm": True})
    fill(page)
    page.locator("#sat-setup-prepare").click()
    page.wait_for_function("typeof window.releaseConfirm === 'function'")
    assert "Press the device reset button" in page.locator("#sat-setup-message").inner_text()
    assert page.evaluate("resetSignals") == [["DTR", False], ["RTS", True], ["RTS", False], ["disconnect"]]
    assert not page.evaluate("calls.some(x => x.method === 'DELETE')")
    page.evaluate("window.releaseConfirm()")
    page.get_by_text("Setup complete: the expected device authenticated successfully.").wait_for()


def test_fallback_shows_all_ports(page):
    page.evaluate(HARNESS, {})
    fill(page)
    page.locator('[name="show_all_ports"]').check()
    page.locator("#sat-setup-prepare").click()
    page.get_by_text("Setup complete: the expected device authenticated successfully.").wait_for()
    assert page.evaluate("portOptions") == {}


def test_serial_open_failure_gives_permissions_help_without_claiming_flash(page):
    page.evaluate(HARNESS, {"openFail": True})
    fill(page)
    page.locator("#sat-setup-prepare").click()
    page.get_by_text("Could not open or connect", exact=False).wait_for()
    text = page.locator("#sat-setup-message").inner_text()
    assert "No firmware was written" in text
    assert "usermod -aG dialout" in text
    assert "log out completely" in text
    assert page.evaluate("flashCount") == 0
    assert page.evaluate("calls.some(x => x.method === 'DELETE')")


def test_validation_failure_does_not_flash(page):
    page.evaluate(HARNESS, {"prepareFail": True})
    fill(page)
    page.locator("#sat-setup-prepare").click()
    page.get_by_text("Invalid bundle", exact=False).wait_for()
    assert page.evaluate("flashCount") == 0
    assert page.locator('[name="wifi_password"]').input_value() == ""


def test_wrong_chip_is_rejected_before_flashing(page):
    page.evaluate(HARNESS, {"wrongChip": True})
    fill(page)
    page.locator("#sat-setup-prepare").click()
    page.get_by_text("This is not an ESP32-S3", exact=False).wait_for()
    assert page.evaluate("flashCount") == 0


def test_cancelled_chooser_is_not_reported_as_a_failed_flash(page):
    page.evaluate(HARNESS, {"selectCancel": True})
    fill(page)
    page.locator("#sat-setup-prepare").click()
    page.get_by_text("No USB device selected", exact=False).wait_for()
    assert page.evaluate("flashCount") == 0
    assert not page.evaluate("calls.some(x => x.url.endsWith('/migration'))")


def test_discovery_and_manual_sonos_override(page):
    page.evaluate(HARNESS, {})
    page.wait_for_function("document.querySelector('#sat-setup-networks option')")
    assert page.locator('#sat-setup-networks option').get_attribute('value') == 'Home Wi-Fi'
    page.locator('summary').click()
    page.locator('[name="music"]').select_option('sonos')
    page.wait_for_function("document.querySelector('[name=sonos_host]').value")
    assert page.locator('[name="sonos_host"]').input_value() == '192.168.1.10'
    page.locator('[name="sonos_host"]').fill('speaker.local')
    assert page.locator('[name="sonos_host"]').input_value() == 'speaker.local'


@pytest.mark.parametrize(
    "options,text",
    [
        ({"secure": False}, "secure context"),
        ({"supported": False}, "WebSerial is unavailable"),
        ({"auth": False}, "Enable a dashboard password"),
    ],
)
def test_unavailable_or_password_gate(page, options, text):
    page.evaluate(HARNESS, options)
    assert text in page.locator("#sat-setup-support").inner_text()
    assert page.locator("#sat-setup-prepare").is_disabled()


@pytest.mark.parametrize("options", [{"fail": True}, {"expire": True}, {"expireAfter": True}])
def test_flash_failure_and_expiry_cancel_transaction(page, options):
    page.evaluate(HARNESS, options)
    fill(page)
    page.locator("#sat-setup-prepare").click()
    page.get_by_text("USB writes cannot be undone", exact=False).last.wait_for()
    assert page.evaluate(
        "calls.some(x => x.method === 'DELETE' && x.url.endsWith('/transactions/abc'))"
    )
    assert page.locator('[name="wifi_password"]').input_value() == ""
    if options.get("expire"):
        assert page.evaluate("flashCount") == 0


def test_cancel_prepared_attempt_and_reopen(page):
    page.evaluate(HARNESS, {"pausePrepare": True})
    fill(page)
    page.locator("#sat-setup-prepare").click()
    page.wait_for_function("typeof window.releasePrepare === 'function'")
    page.locator("#sat-setup-cancel").click()
    assert page.evaluate("calls.some(x => x.method === 'DELETE')")
    assert "USB writes cannot be undone" in page.locator("#sat-setup-message").inner_text()
    page.evaluate("window.releasePrepare()")
    page.wait_for_function("!document.querySelector('#sat-setup-fields').disabled")
    page.locator("#sat-setup-cancel").click()
    page.evaluate("wizard.open()")
    assert page.locator("#sat-setup").is_visible()


def test_cancel_during_flash_revokes_transaction(page):
    page.evaluate(HARNESS, {"pauseFlash": True})
    fill(page)
    page.locator("#sat-setup-prepare").click()
    page.wait_for_function("typeof window.releaseFlash === 'function'")
    page.locator("#sat-setup-cancel").click()
    assert page.evaluate("calls.some(x => x.method === 'DELETE')")
    page.evaluate("window.releaseFlash()")
    assert "Setup cancelled" in page.locator("#sat-setup-message").inner_text()
    assert not page.evaluate(
        "calls.some(x => x.url.endsWith('/transactions/abc') && x.method === 'GET' && calls.indexOf(x) > calls.findIndex(y => y.method === 'DELETE'))"
    )
