"""Standalone real-DOM ES-module integration checks (no model/server startup).

Run: .venv/bin/python tests/browser_js_integration.py
Requires Playwright and its Chromium browser. HTTP, SSE, WebSocket and audio
devices are controlled at browser boundaries; production modules and HTML run
unchanged, with no bundler or globals exported for tests.
"""
import json
import sys
import unittest
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.config_schema import schema_as_dicts, tier_presets_as_dicts  # noqa: E402

STATIC = ROOT / "server/static"

BOUNDARIES = """
window.FULLOCH_DASHBOARD_PREFS = {theme: 'auto', show_turn_details: false};
window.boundary = {streams: [], sockets: [], contexts: [], tracks: [], nodes: [],
  sources: [], previews: [], revoked: [], intervals: new Set(), micDelay: false};
const b = window.boundary;
const interval = window.setInterval.bind(window), clear = window.clearInterval.bind(window);
window.setInterval = (...args) => { const id = interval(...args); b.intervals.add(id); return id; };
window.clearInterval = id => { b.intervals.delete(id); clear(id); };
const revoke = URL.revokeObjectURL.bind(URL);
URL.revokeObjectURL = url => { b.revoked.push(url); revoke(url); };
window.EventSource = class {
  constructor(url) { this.url = url; b.streams.push(this); queueMicrotask(() => this.onopen?.()); }
  emit(event) { this.onmessage?.({data: JSON.stringify(event)}); }
  close() { this.closed = true; }
};
window.WebSocket = class {
  static OPEN = 1; static CONNECTING = 0; static CLOSED = 3;
  constructor(url) {
    this.url = url; this.readyState = 0; this.sent = []; b.sockets.push(this);
    queueMicrotask(() => { if (this.readyState !== 0) return; this.readyState = 1; this.onopen?.(); });
  }
  send(data) { this.sent.push(data); }
  emit(data) { this.onmessage?.({data: typeof data === 'object' && !(data instanceof ArrayBuffer) ? JSON.stringify(data) : data}); }
  close() { this.readyState = 3; queueMicrotask(() => this.onclose?.()); }
};
window.AudioContext = class {
  constructor() {
    this.currentTime = 0; this.destination = {}; b.contexts.push(this);
    this.audioWorklet = {addModule: async () => {
      if (b.workletFails) throw new Error('worklet failure');
    }};
  }
  resume() { return Promise.resolve(); }
  close() { this.closed = true; return Promise.resolve(); }
  createMediaStreamSource() { return {connect() {}}; }
  createBuffer(channels, length, rate) {
    const samples = new Float32Array(length);
    return {duration: length / rate, getChannelData: () => samples};
  }
  createBufferSource() {
    const node = {connect() {}, start(at) { this.at = at; }, stop() { this.stopped = true; }};
    b.sources.push(node); return node;
  }
};
window.AudioWorkletNode = class {
  constructor() { this.port = {close() { this.closed = true; }}; b.nodes.push(this); }
  disconnect() { this.disconnected = true; }
};
Object.defineProperty(navigator, 'mediaDevices', {value: {
  getUserMedia: async () => {
    const track = {stop() { this.stopped = true; }, applyConstraints: async () => {}};
    b.tracks.push(track);
    const stream = {getTracks: () => [track], getAudioTracks: () => [track]};
    if (b.micDelay) await new Promise(resolve => { b.releaseMic = resolve; });
    return stream;
  }
}});
window.Audio = class {
  constructor(url) { this.url = url; this.paused = true; b.previews.push(this); }
  play() { this.paused = false; return Promise.resolve(); }
  pause() { this.paused = true; }
};
"""


class BrowserModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        self.page = self.browser.new_page()
        self.page.set_default_timeout(5000)
        self.errors = []
        self.requests = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.add_init_script(BOUNDARIES)
        fields = schema_as_dicts()
        for field in fields:
            field.update(value=field["default"], set=False)
        self.schema = {
            "fields": fields, "models": None, "credentials": {}, "variant": "cpu",
            "tier_presets": tier_presets_as_dicts(),
            "wakeword_presets": [{"wakeword": "hey atticus", "pattern": "atticus",
                                  "model": "", "label": "Hey Atticus", "recommended": True}],
            "backends": {
                domain: [{"backend": name, "offerable": True, "display_name": name}
                         for name in names]
                for domain, names in {
                    "asr": ["qwen-onnx", "qwen"],
                    "tts": ["pocket-tts-onnx", "kokoro-onnx", "qwen"],
                    "llm": ["none", "external", "local"],
                }.items()
            },
        }
        self.status = {"phase": "READY", "state": "idle", "server_instance_id": "test"}
        self.progress = {"state": "downloading", "assets": []}
        self.history = [{"role": "assistant", "source": "startup", "ts": 1, "content": "Hello"}]
        self.page.route("**/*", self.route)

    def tearDown(self):
        self.page.close()
        self.assertEqual(self.errors, [], "Uncaught browser errors")

    def route(self, route):
        path = urlparse(route.request.url).path
        if path in ("/", "/setup"):
            name = "setup" if path == "/setup" else "index"
            return route.fulfill(path=STATIC / f"{name}.html", content_type="text/html")
        if path.startswith("/static/"):
            file = STATIC / path.removeprefix("/static/")
            mime = "text/javascript" if file.suffix == ".js" else "text/css"
            return route.fulfill(path=file, content_type=mime)
        if path == "/logo.png":
            return route.fulfill(status=204)
        self.requests.append((path, route.request.method, route.request.post_data))
        data = {
            "/status": self.status, "/history": self.history,
            "/config": {"wakeword": "hey atticus", "restart_required": False},
            "/api/obsidian/status": {}, "/api/obsidian/show-token": {"token": "test-token"},
            "/ha/areas": {"available": True, "areas": [{"id": "kitchen", "name": "Kitchen"}]},
            "/setup/schema": self.schema, "/setup/preflight": {},
            "/setup/voices": {"voices": ["atticus", "other"]},
            "/setup/progress": self.progress, "/setup/backups": {"backups": []},
            "/setup/voice/save": {"saved": "other"},
        }.get(path, {"ok": True})
        route.fulfill(body=json.dumps(data), content_type="application/json")

    def open_chat(self):
        self.page.goto("http://localhost/")
        self.page.wait_for_function("boundary.streams.length === 1")

    def emit(self, event):
        self.page.evaluate("ev => boundary.streams.at(-1).emit(ev)", event)

    def test_chat_history_trace_cards_stats_commands_and_reset(self):
        self.open_chat()
        self.assertEqual(self.page.locator(".msg.assistant .bubble").first.inner_text(), "Hello")
        self.page.locator("#input").fill("What is playing?")
        self.page.locator("#input").press("Enter")
        self.page.wait_for_function("document.querySelector('#send').title === 'Stop'")
        self.emit({"role": "user", "source": "text", "ts": 2, "content": "What is playing?"})
        artifact = {"type": "media", "title": "Track", "player": "Kitchen", "state": "playing"}
        self.emit({"role": "agent", "kind": "observation", "ts": 3,
                   "payload": {"intent": "media", "result": "playing", "artifact": artifact}})
        self.emit({"role": "assistant", "source": "text", "ts": 4, "content": "Playing Track",
                   "stats": {"total": 1.0}})
        self.assertEqual(self.page.locator(".trace-group summary").inner_text(), "trace · 1 event")
        self.assertEqual(self.page.locator(".media-card").count(), 1)
        self.emit({"role": "stats", "ref_ts": 4, "patch": {"total": 2.5}})
        self.assertEqual(self.page.locator(".stats-btn").inner_text(), "2.50s")
        self.page.get_by_role("button", name="pause Kitchen", exact=True).click()
        self.page.wait_for_timeout(30)
        self.assertTrue(any(path == "/chat" and 'pause Kitchen' in (body or '')
                            for path, _, body in self.requests))
        self.emit({"role": "stopped"})
        self.assertEqual(self.page.locator("#send").get_attribute("title"), "Send")
        timers = {"type": "timers", "timers": [{"id": "t1", "duration": 60, "ends_at": 9999999999}]}
        self.emit({"role": "assistant", "source": "text", "ts": 5, "content": "Timer", "artifact": timers})
        before = self.page.evaluate("boundary.intervals.size")
        self.emit({"role": "reset"})
        self.assertEqual(self.page.evaluate("boundary.intervals.size"), before - 1)
        self.assertEqual(self.page.locator("#empty").count(), 1)
        self.emit({"role": "assistant", "source": "text", "ts": 6, "content": "After reset"})
        self.assertEqual(self.page.locator("#empty").count(), 0)
        self.page.evaluate("window.dispatchEvent(new Event('pagehide'))")
        self.assertTrue(self.page.evaluate("boundary.streams.every(s => s.closed) && boundary.intervals.size === 0"))

    def test_satellite_replay_capture_cancel_and_teardown(self):
        self.open_chat()
        self.page.locator(".msg-play").click()
        self.page.wait_for_function("boundary.sockets.length === 1")
        self.assertEqual(self.page.evaluate("boundary.tracks.length"), 0)
        self.page.locator("#satellite-btn").click()  # Disconnect replay.
        self.page.locator("#satellite-btn").click()  # Connect microphone.
        self.page.wait_for_function("boundary.nodes.length === 1 && boundary.sockets.length === 2")
        self.page.evaluate("""() => {
          const ws = boundary.sockets.at(-1);
          ws.emit({type: 'session', satellite_id: 'mine', half_duplex: true});
          boundary.nodes[0].port.onmessage({data: new Float32Array(3200).buffer});
          ws.emit({type: 'tts_start', sr: 24000});
          ws.emit(new Float32Array(6000).buffer);
        }""")
        self.assertEqual(self.page.evaluate("boundary.sources.length"), 1)
        self.page.evaluate("boundary.sockets.at(-1).emit({type: 'tts_cancel'})")
        self.assertTrue(self.page.evaluate("boundary.sources[0].stopped"))
        self.page.evaluate("boundary.sockets.at(-1).emit(new Float32Array(6000).buffer)")
        self.assertEqual(self.page.evaluate("boundary.sources.length"), 1)
        self.page.evaluate("window.dispatchEvent(new Event('pagehide'))")
        self.assertTrue(self.page.evaluate("""boundary.contexts.every(c => c.closed) &&
            boundary.tracks.every(t => t.stopped) && boundary.nodes.every(n => n.port.closed) &&
            boundary.sockets.every(s => s.readyState === 3) && boundary.intervals.size === 0"""))

    def test_satellite_cancel_pending_mic_and_worklet_failure(self):
        self.open_chat()
        self.page.evaluate("boundary.micDelay = true")
        self.page.locator("#satellite-btn").click()
        self.page.wait_for_function("!!boundary.releaseMic")
        self.page.locator("#satellite-btn").click()
        self.page.evaluate("boundary.releaseMic()")
        self.page.wait_for_function("boundary.tracks[0].stopped")
        self.assertEqual(self.page.evaluate("boundary.sockets.length"), 0)
        self.page.evaluate("boundary.micDelay = false; boundary.workletFails = true")
        self.page.locator("#satellite-btn").click()
        self.page.wait_for_function("boundary.tracks.length === 2 && boundary.tracks[1].stopped")
        self.assertTrue(self.page.evaluate("boundary.contexts.every(c => c.closed)"))

    def test_wizard_navigation_voice_install_progress_and_public_retry(self):
        self.status["phase"] = "NEEDS_SETUP"
        self.page.goto("http://localhost/setup")
        self.page.locator("#next1").click()
        self.page.locator("#voice-sel").select_option("other")
        self.page.locator(".voice-play").click()
        self.assertFalse(self.page.evaluate("boundary.previews[0].paused"))
        self.page.locator("#get-started").click()
        self.assertTrue(self.page.evaluate("boundary.previews[0].paused"))
        self.page.locator("#connect-skip").click()
        self.progress.update(state="error", error="Access required", needs_hf_token=True)
        self.page.locator("#obsidian-skip").click()
        self.page.locator("#hf-token").fill("hf_test")
        self.page.get_by_role("button", name="Save token and retry").click()
        self.page.wait_for_timeout(50)
        self.assertTrue(any(path == "/setup/retry-download" for path, _, _ in self.requests))
        models = [json.loads(body)["models"] for path, _, body in self.requests if path == "/setup/models"]
        self.assertEqual(models[0]["llm"]["backend"], "none")
        configs = [json.loads(body)["updates"] for path, method, body in self.requests
                   if path == "/config" and method == "PUT"]
        self.assertEqual(configs[0]["general.voice_clone"], "other")
        self.progress.update(state="done")
        self.status.update(phase="READY")
        self.page.get_by_role("button", name="Retry download", exact=True).click()
        self.page.locator("#finish-name").wait_for()
        self.assertTrue(self.page.get_by_text("Almost done", exact=True).is_visible())

    def test_settings_cards_fields_and_model_save(self):
        self.schema["models"] = self.schema["tier_presets"][2]["models"]
        self.page.goto("http://localhost/setup")
        self.page.locator("#save-cfg").wait_for()
        self.page.wait_for_function("document.querySelector('#sec-obs-token')?.textContent === 'test-token'")
        self.page.locator("details.settings-card").evaluate_all("nodes => nodes.forEach(n => n.open = true)")
        self.page.locator("#sm-llama-sel").select_option("external")
        self.page.locator("#sm-oai-url").fill("localhost:11434")
        self.page.locator("#sm-oai-model").fill("test-model")
        self.page.locator("#save-models").click()
        self.page.wait_for_timeout(50)
        models = [json.loads(body)["models"] for path, _, body in self.requests if path == "/setup/models"]
        self.assertEqual(models[0]["llm"]["base_url"], "http://localhost:11434/v1")
        self.assertEqual(models[0]["llm"]["model"], "test-model")
        self.page.locator("#cf-general_voice_clone").select_option("other")
        self.page.locator("#save-cfg").click()
        self.page.wait_for_timeout(50)
        self.assertTrue(any(path == "/config" and 'other' in (body or '') for path, _, body in self.requests))

    def test_all_artifact_dispatches_and_safe_text_links(self):
        self.open_chat()
        report = {"title": "Report", "summary": "Summary", "created_at": 1,
                  "report_url": "/reports/fulloch-reports/2026-09-11-1234abcd"}
        artifacts = [
            {"type": "weather", "title": "<script>bad()</script>", "current": {"temperature": 20},
             "forecast": [{"label": "Today", "condition": "rain", "high": 22, "low": 10}]},
            {"type": "entity_status", "title": "Lamp", "domain": "light", "state": "on"},
            {"type": "temperature_history", "points": [{"value": 20}], "current": 20, "min": 10, "max": 22},
            {"type": "light_history", "points": [{"on": True, "brightness": 50}], "state": "on"},
            {"type": "media", "title": "Song", "player": "Kitchen"},
            {"type": "calendar", "events": [{"title": "Meeting", "when": "Tomorrow"}]},
            {"type": "timers", "timers": [{"id": "t", "duration": 60, "ends_at": 9999999999}]},
            {"type": "home_overview", "groups": [{"label": "Lights", "kind": "lights", "count": 1, "entities": ["Lamp"]}]},
            {"type": "energy", "metrics": [{"kind": "solar", "label": "Solar", "value": 3}]},
            {"type": "security", "status": "secure", "groups": []},
            {"type": "note", "title": "Note", "excerpt": "<b>literal</b>"},
            {"type": "notes_search", "query": "Note", "matches": [{"title": "Note", "excerpt": "Text"}]},
            {"type": "todos", "items": ["Buy milk"]},
            {"type": "web_research", "query": "Topic", "sources": [
                {"url": "javascript:bad()", "host": "bad", "evidence": "Ignored"},
                {"url": "https://example.com", "host": "Example", "evidence": "Text"}]},
            {"type": "finance_exchange_rate", "exchange_rate": {"base_currency": "USD", "quote_currency": "EUR", "rate": 0.9}},
            {"type": "finance_quote", "quote": {"name": "Stock", "price": "123"}},
            *[{"type": kind, **report} for kind in ["research_report", "travel_report", "finance_report", "generated_report"]],
            {"type": "flight_plan", "route": {"origin": "A", "destination": "B"}, "offer": {"price": 123}},
        ]
        for ts, artifact in enumerate(artifacts, 2):
            self.emit({"role": "assistant", "source": "text", "ts": ts, "content": "Result", "artifact": artifact})
        self.assertEqual(self.page.locator(".artifact").count(), len(artifacts))
        self.assertEqual(self.page.locator(".artifact script, .note-card p b, a[href^='javascript:']").count(), 0)
        self.assertIn("<script>bad()</script>", self.page.locator(".weather-title").inner_text())
        self.assertEqual(self.page.locator(".web-source a").get_attribute("rel"), "noopener noreferrer")

    def test_install_ignores_stale_progress_and_destroyed_loading(self):
        self.open_chat()
        result = self.page.evaluate("""async () => {
          const {createInstallProgress} = await import('/static/js/setup-install.js');
          document.body.innerHTML = '<div id="screen"></div><div id="steps"></div><div id="subtitle"></div><div id="alert"></div>';
          const fetchBefore = window.fetch;
          const pending = [];
          window.fetch = () => new Promise(resolve => pending.push(resolve));
          const state = {sel: {}, cameViaWizard: false};
          let finished = 0;
          const component = createInstallProgress({state, chosenModels: () => ({}),
            boot: () => finished++, stepFinish: () => finished++});
          component.showProgress();
          component.showLoading();
          // The old progress response arrives after the loading view replaced it.
          pending.shift()(new Response(JSON.stringify({state: 'error', assets: []})));
          await new Promise(resolve => setTimeout(resolve, 0));
          const loading = !!document.querySelector('#term');
          component.destroy();
          pending.shift()(new Response(JSON.stringify({phase: 'READY'})));
          await new Promise(resolve => setTimeout(resolve, 0));
          window.fetch = fetchBefore;
          return {loading, finished};
        }""")
        self.assertEqual(result, {"loading": True, "finished": 0})

    def test_generated_voice_url_revoked_when_leaving_step(self):
        self.status["phase"] = "NEEDS_SETUP"
        self.page.goto("http://localhost/setup")
        self.page.locator("#next1").click()
        self.page.locator("#gen-voice").click()
        self.page.locator("#gv-instruct").fill("A warm voice")
        self.page.locator("#gv-generate").click()
        self.page.wait_for_function("document.querySelector('#gv-status')?.textContent === 'Preview ready.'")
        url = self.page.locator("#gv-audio audio").get_attribute("src")
        self.page.locator("#back2").click()
        self.assertTrue(self.page.evaluate("url => boundary.revoked.includes(url)", url))


if __name__ == "__main__":
    unittest.main(verbosity=2)
