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
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.backends import TTS_MODEL_LANGUAGES  # noqa: E402
from server.config_schema import (  # noqa: E402
    schema_as_dicts,
    tier_presets_as_dicts,
    wakeword_presets_as_dicts,
)
from server.config_store import _backends_view  # noqa: E402
from utils.locale import intent_languages  # noqa: E402

STATIC = ROOT / "server/static"

BOUNDARIES = """
window.FULLOCH_DASHBOARD_PREFS = {theme: 'auto', show_turn_details: false};
window.boundary = {streams: [], sockets: [], contexts: [], tracks: [], nodes: [],
  sources: [], previews: [], revoked: [], logoFrames: [], intervals: new Set(), micDelay: false};
const b = window.boundary;
const drawImage = CanvasRenderingContext2D.prototype.drawImage;
CanvasRenderingContext2D.prototype.drawImage = function(...args) {
  if (this.canvas.id === 'brand-logo') {
    b.logoSheet = args[0];
    b.logoFrames.push(args[2] / 314 * 4 + args[1] / 314);
  }
  return drawImage.apply(this, args);
};
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
    this.currentTime = 0; this.state = 'suspended'; this.destination = {}; b.contexts.push(this);
    this.audioWorklet = {addModule: async () => {
      if (b.workletFails) throw new Error('worklet failure');
    }};
  }
  resume() { this.state = 'running'; this.onstatechange?.(); return Promise.resolve(); }
  suspend() { this.state = 'suspended'; this.onstatechange?.(); return Promise.resolve(); }
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
            "intent_languages": intent_languages(),
            "tts_model_languages": {key: list(value) for key, value in TTS_MODEL_LANGUAGES.items()},
            "default_backends": {"asr": "moonshine", "tts": "kokoro-onnx", "llm": "laya"},
            "tier_presets": tier_presets_as_dicts(),
            "wakeword_presets": wakeword_presets_as_dicts(),
            "backends": _backends_view(),
        }
        for backends in self.schema["backends"].values():
            for backend in backends:
                backend["offerable"] = not backend["gpu_only"]
        self.status = {"phase": "READY", "state": "idle", "server_instance_id": "test"}
        self.documents = {"documents": [{"path": "Fulloch/example-report.md", "title": "example report"}]}
        self.progress = {"state": "downloading", "assets": []}
        self.intent_prepare_error = False
        self.config_error = False
        self.history = [{"role": "assistant", "source": "startup", "ts": 1, "content": "Hello"}]
        self.entities = None
        self.entity_update_error = False
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
        if path == "/fulloch_animate.png":
            return route.fulfill(path=ROOT / "utils/fulloch_animate.png", content_type="image/png")
        self.requests.append((path, route.request.method, route.request.post_data))
        if self.entities is not None and path in ("/entities", "/entities/bulk"):
            if route.request.method == "POST":
                if self.entity_update_error:
                    return route.fulfill(status=500, body="{}", content_type="application/json")
                body = json.loads(route.request.post_data)
                ids = body.get("entity_ids", [body.get("entity_id")])
                for entity in self.entities:
                    if entity["entity_id"] in ids:
                        entity["denied"] = body["denied"]
            return route.fulfill(body=json.dumps({"available": True, "entities": self.entities}),
                                 content_type="application/json")
        if (path == "/setup/intent-languages" and self.intent_prepare_error) or (
            path == "/config" and route.request.method == "PUT" and self.config_error
        ):
            return route.fulfill(status=503, body=json.dumps({"detail": "Download failed"}), content_type="application/json")
        data = {
            "/status": self.status, "/history": self.history,
            "/config": {"wakeword": "hey atticus", "restart_required": False},
            "/api/obsidian/status": {}, "/api/obsidian/show-token": {"token": "test-token"},
            "/api/documents": self.documents,
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

    def test_entity_groups_bulk_search_and_failure(self):
        self.entities = [
            {"entity_id": "light.a", "name": "Desk", "domain": "light", "denied": False},
            {"entity_id": "light.b", "name": "Hall", "domain": "light", "denied": False},
            {"entity_id": "lock.door", "name": "Door", "domain": "lock", "denied": True},
        ]
        self.open_chat()
        self.page.get_by_role("tab", name="Entities").click()
        groups = self.page.locator(".entity-group")
        self.page.wait_for_function("document.querySelectorAll('.entity-group').length === 2")
        self.assertEqual(groups.evaluate_all("els => els.map(e => e.open)"), [False, False])
        self.page.locator(".entity-group summary").first.focus()
        self.page.keyboard.press("Enter")
        self.assertTrue(groups.first.evaluate("el => el.open"))

        self.page.locator("#entity-search").fill("Desk")
        self.assertTrue(groups.first.evaluate("el => el.open"))
        self.page.get_by_role("button", name="Block all light entities").click()
        self.page.wait_for_function("document.querySelector('#entities-allow-all').disabled === false && "
                                    "document.querySelector('[aria-label=\"Block all light entities\"]').disabled")
        self.assertTrue(self.entities[0]["denied"])
        self.assertTrue(self.entities[1]["denied"])
        self.assertTrue(self.entities[2]["denied"])
        request = [json.loads(body) for path, method, body in self.requests
                   if path == "/entities/bulk" and method == "POST"][-1]
        self.assertEqual(set(request["entity_ids"]), {"light.a", "light.b"})

        self.page.locator("#entities-allow-all").click()
        self.page.wait_for_function("document.querySelector('#entities-allow-all').disabled && "
                                    "!document.querySelector('#entities-block-all').disabled")
        self.assertTrue(all(not entity["denied"] for entity in self.entities))
        self.assertTrue(groups.first.evaluate("el => el.open"))

        self.page.locator('[aria-label="Allow voice control of Desk"]').evaluate(
            "el => { el.checked = false; el.dispatchEvent(new Event('change', {bubbles: true})); }"
        )
        self.page.wait_for_function("!document.querySelector('[aria-label=\"Allow all light entities\"]').disabled")
        self.assertTrue(self.entities[0]["denied"])
        self.page.get_by_role("button", name="Allow all light entities").click()
        self.page.wait_for_function("document.querySelector('[aria-label=\"Allow all light entities\"]').disabled && "
                                    "!document.querySelector('[aria-label=\"Block all light entities\"]').disabled")
        self.assertTrue(all(not entity["denied"] for entity in self.entities))

        self.entity_update_error = True
        self.page.locator("#entities-block-all").click()
        self.page.get_by_role("alert").wait_for(state="visible")
        self.assertTrue(all(not entity["denied"] for entity in self.entities))
        self.assertTrue(self.page.locator('[aria-label="Allow voice control of Desk"]').is_checked())
        self.assertTrue(groups.first.evaluate("el => el.open"))
        self.entity_update_error = False
        self.page.reload()
        self.page.get_by_role("tab", name="Entities").click()
        self.page.wait_for_function("document.querySelectorAll('.entity-group').length === 2")
        self.assertEqual(groups.evaluate_all("els => els.map(e => e.open)"), [False, False])
        groups.first.locator("summary").click()
        self.assertTrue(self.page.locator('[aria-label="Allow voice control of Desk"]').is_checked())

    def test_conversation_menu_save_list_search_load_delete_and_clear(self):
        self.history = [{"role": "user", "source": "text", "ts": 1, "content": "Current chat"}]
        snapshots = [{"id": f"id-{i}", "title": f"Chat {i}", "created_at": 1700000000 + i} for i in range(25)]

        def route_conversation(route):
            path = urlparse(route.request.url).path
            if path == "/conversations/status":
                return route.fulfill(json={"persisted": True})
            if path == "/conversations" and route.request.method == "POST":
                return route.fulfill(json={"id": "new-id", "created_at": 1700000000})
            if path == "/conversations" or path == "/conversations/search":
                offset = int(parse_qs(urlparse(route.request.url).query).get("offset", [0])[0])
                if path.endswith("search"):
                    offset = route.request.post_data_json["offset"]
                    entries = [item for item in snapshots if item["id"] == "id-3"]
                else:
                    entries = snapshots
                return route.fulfill(json={"conversations": entries[offset:offset + 20],
                                           "next_offset": offset + 20 if offset + 20 < len(entries) else None})
            if path.endswith("/load"):
                self.assertEqual(route.request.post_data_json["current"], "save")
                self.history = [{"role": "user", "source": "text", "ts": 4, "content": "Restored chat"}]
                return route.fulfill(json={"ok": True, "saved": {"id": "saved-current", "created_at": 1700000000}})
            if route.request.method == "DELETE":
                snapshots[:] = [item for item in snapshots if item["id"] != path.rsplit("/", 1)[-1]]
                return route.fulfill(json={"ok": True})
            return route.fulfill(status=404)

        self.page.route("**/conversations**", route_conversation)
        self.open_chat()
        self.page.get_by_text("Current chat", exact=True).wait_for()
        button = self.page.get_by_role("button", name="Conversation options")
        button.click()
        self.page.get_by_role("button", name="Save conversation", exact=True).click()
        self.page.locator("#conversation-status").wait_for(state="visible")
        self.assertIn("Conversation saved.", self.page.locator("#conversation-status").inner_text())
        self.assertIn("new-id", self.page.locator("#conversation-status").inner_text())
        button.click()
        self.page.get_by_role("button", name="Saved conversations", exact=True).click()
        self.page.wait_for_function("document.querySelectorAll('.conversation-item').length === 20")
        self.page.locator("#conversation-list").evaluate("el => { el.scrollTop = el.scrollHeight; }")
        self.page.wait_for_function("document.querySelectorAll('.conversation-item').length === 25")
        search = self.page.get_by_role("searchbox", name="Search saved conversations")
        search.fill("garden")
        search.press("Enter")
        self.page.wait_for_function("document.querySelectorAll('.conversation-item').length === 1")
        self.page.once("dialog", lambda dialog: dialog.accept())
        self.page.get_by_role("button", name="Delete saved conversation Chat 3").click()
        self.page.get_by_text("No saved conversations found.").wait_for()
        self.page.get_by_role("button", name="Clear", exact=True).click()
        self.page.wait_for_function("document.querySelectorAll('.conversation-item').length === 20")
        self.page.once("dialog", lambda dialog: dialog.accept())
        self.page.locator(".conversation-load").first.click()
        self.page.get_by_text("Restored chat", exact=True).wait_for()
        self.assertIn("saved-current", self.page.locator("#conversation-status").inner_text())
        button.click()
        self.page.keyboard.press("Escape")
        self.assertTrue(self.page.locator("#conversation-menu").is_hidden())
        button.click()
        self.page.once("dialog", lambda dialog: dialog.accept())
        self.page.get_by_role("button", name="Clear conversation", exact=True).click()
        self.page.get_by_text("Conversation cleared.").wait_for()
        self.assertLess(button.bounding_box()["y"] - self.page.locator("#chat").bounding_box()["y"], 30)

    def test_protocol_events_do_not_render_in_history_or_live_chat(self):
        self.history.append({"type": "assistant.stand_down_after_tts", "satellite_id": "kitchen"})
        self.open_chat()
        self.page.get_by_role("button", name="Kitchen", exact=True).click()
        self.page.wait_for_function("document.querySelectorAll('.msg.assistant').length === 1")
        self.emit({"type": "assistant.stand_down_after_tts", "satellite_id": "kitchen"})
        self.emit({"role": "assistant", "source": "proactive", "ts": 2, "content": "Reminder"})
        self.page.wait_for_function("document.querySelectorAll('.msg.assistant').length === 2")
        self.assertEqual(self.page.locator('.msg.undefined').count(), 0)
        self.assertEqual(self.page.locator('.msg.assistant .bubble').all_text_contents(),
                         ["Hello", "Reminder"])

    def test_thinking_close_confirmation_and_next_job(self):
        job = {"id": "first", "status": "RUNNING", "task": "First job"}
        self.status["thinking_job"] = job
        self.open_chat()
        self.page.wait_for_function("document.querySelector('#thinking-job-task').textContent === 'First job'")
        self.page.once("dialog", lambda dialog: dialog.dismiss())
        self.page.locator("#thinking-job-cancel").click()
        self.assertTrue(self.page.locator("#thinking-job-card").is_visible())
        self.assertFalse(any(path == "/thinking/first/cancel" for path, _, _ in self.requests))

        self.page.once("dialog", lambda dialog: dialog.accept())
        self.page.locator("#thinking-job-cancel").click()
        self.page.wait_for_function("document.querySelector('#thinking-job-card').hidden")
        self.assertTrue(any(path == "/thinking/first/cancel" and method == "POST"
                            for path, method, _ in self.requests))
        # A stale status must not resurrect the cancelled card.
        self.page.wait_for_timeout(1200)
        self.assertTrue(self.page.locator("#thinking-job-card").is_hidden())
        self.status["thinking_job"] = {"id": "second", "status": "RUNNING", "task": "Next job"}
        self.page.wait_for_function("document.querySelector('#thinking-job-task').textContent === 'Next job'")

    def test_thinking_close_finished_job_keeps_report(self):
        self.status["thinking_job"] = {"id": "done", "status": "READY", "task": "Finished",
                                       "partial_artifact_link": "/reports/Fulloch/example-report.md"}
        self.open_chat()
        self.page.wait_for_function("document.querySelector('#thinking-job-task').textContent === 'Finished'")
        self.page.locator("#thinking-job-cancel").click()
        self.assertTrue(self.page.locator("#thinking-job-card").is_hidden())
        self.assertFalse(any(path == "/thinking/done/cancel" for path, _, _ in self.requests))
        self.assertEqual(len(self.documents["documents"]), 1)
        self.page.wait_for_timeout(1200)
        self.assertTrue(self.page.locator("#thinking-job-card").is_hidden())
        self.page.reload()
        self.page.wait_for_function("boundary.streams.length === 1")
        self.assertTrue(self.page.locator("#thinking-job-card").is_hidden())

    def test_thinking_terminal_job_expires_and_stays_hidden(self):
        self.page.clock.install()
        self.status["thinking_job"] = {"id": "failed", "status": "FAILED", "task": "Failed job"}
        self.open_chat()
        self.page.wait_for_function("document.querySelector('#thinking-job-task').textContent === 'Failed job'")
        self.page.clock.run_for(299000)
        self.assertTrue(self.page.locator("#thinking-job-card").is_visible())
        self.page.clock.run_for(1100)
        self.assertTrue(self.page.locator("#thinking-job-card").is_hidden())
        self.page.reload()
        self.page.wait_for_function("boundary.streams.length === 1")
        self.assertTrue(self.page.locator("#thinking-job-card").is_hidden())

    def test_old_finished_job_is_not_shown(self):
        self.status["thinking_job"] = {"id": "old", "status": "READY", "task": "Old job", "finished_at": 1}
        self.open_chat()
        self.page.wait_for_timeout(200)
        self.assertTrue(self.page.locator("#thinking-job-card").is_hidden())

    def test_brand_subtitle_wraps_on_narrow_screen(self):
        self.page.set_viewport_size({"width": 320, "height": 700})
        self.open_chat()
        subtitle = self.page.locator("#brand-sub")
        self.assertEqual(subtitle.evaluate("el => getComputedStyle(el).whiteSpace"), "normal")
        self.assertGreater(subtitle.evaluate("el => el.getBoundingClientRect().height"),
                           subtitle.evaluate("el => parseFloat(getComputedStyle(el).fontSize) * 1.5"))

    def test_logo_idle_blinks_glances_and_teardown(self):
        self.page.clock.install()
        self.page.add_init_script("Math.random = () => 0.5;")
        self.open_chat()
        self.page.wait_for_function("boundary.logoFrames.length > 0")
        self.assertEqual(self.page.locator("header .logo").evaluate(
            "el => [el.clientWidth, el.clientHeight]"), [80, 80])
        self.assertEqual(self.page.locator("#brand-logo").evaluate("el => el.tagName"), "CANVAS")
        self.assertEqual(self.page.evaluate("boundary.logoFrames[0]"), 0)
        self.page.clock.run_for(4600)
        self.assertEqual(self.page.evaluate("boundary.logoFrames.slice(-4)"), [1, 2, 1, 0])
        self.page.evaluate("Math.random = () => 0.1")
        self.page.clock.run_for(6000)
        self.assertIn(4, self.page.evaluate("boundary.logoFrames"))
        self.page.evaluate("window.dispatchEvent(new Event('pagehide'))")
        draws = self.page.evaluate("boundary.logoFrames.length")
        self.page.clock.run_for(10000)
        self.assertEqual(self.page.evaluate("boundary.logoFrames.length"), draws)

    def test_logo_body_pixels_remain_identical_in_every_frame(self):
        self.open_chat()
        self.page.wait_for_function("boundary.logoFrames.length > 0")
        changes = self.page.evaluate("""() => {
          const context = boundary.logoSheet.getContext('2d');
          const idle = context.getImageData(0, 0, 314, 314).data;
          const faceRegions = [[120, 94, 170, 144], [215, 94, 263, 144], [163, 120, 222, 171]];
          return Array.from({length: 16}, (_, frame) => {
            const pixels = context.getImageData(frame % 4 * 314, Math.floor(frame / 4) * 314, 314, 314).data;
            let body = 0, face = 0;
            for (let y = 0; y < 314; y++) {
              for (let x = 0; x < 314; x++) {
                const i = (y * 314 + x) * 4;
                if (!pixels.subarray(i, i + 4).some((value, channel) => value !== idle[i + channel])) continue;
                if (faceRegions.some(([left, top, right, bottom]) =>
                    x >= left && x < right && y >= top && y < bottom)) face++;
                else body++;
              }
            }
            return {body, face};
          });
        }""")
        self.assertTrue(all(frame["body"] == 0 for frame in changes), changes)
        self.assertTrue(all(changes[frame]["face"] > 0 for frame in [1, 2, 4, 5, 6, 7, *range(8, 16)]))

    def test_logo_talking_tracks_playback_not_generation(self):
        self.open_chat()
        self.page.wait_for_function("boundary.logoFrames.length > 0")
        self.page.locator(".msg-play").click()
        self.page.wait_for_function("boundary.sockets.length === 1")
        self.page.evaluate("boundary.sockets[0].emit({type: 'tts_start', sr: 24000})")
        self.assertEqual(self.page.evaluate("boundary.logoFrames.at(-1)"), 0)
        self.page.evaluate("boundary.sockets[0].emit(new Float32Array(6000).buffer)")
        self.assertGreaterEqual(self.page.evaluate("boundary.logoFrames.at(-1)"), 8)
        self.page.evaluate("boundary.sockets[0].emit({type: 'tts_end'})")
        self.assertGreaterEqual(self.page.evaluate("boundary.logoFrames.at(-1)"), 8)
        self.page.evaluate("boundary.contexts[0].suspend()")
        self.assertEqual(self.page.evaluate("boundary.logoFrames.at(-1)"), 0)
        self.page.evaluate("boundary.contexts[0].resume()")
        self.assertGreaterEqual(self.page.evaluate("boundary.logoFrames.at(-1)"), 8)
        self.page.evaluate("boundary.sources[0].onended()")
        self.assertEqual(self.page.evaluate("boundary.logoFrames.at(-1)"), 0)
        self.page.evaluate("""() => {
          boundary.sockets[0].emit({type: 'tts_start'});
          boundary.sockets[0].emit(new Float32Array(6000).buffer);
          boundary.sockets[0].emit({type: 'tts_cancel'});
        }""")
        self.assertEqual(self.page.evaluate("boundary.logoFrames.at(-1)"), 0)
        self.assertTrue(self.page.evaluate("boundary.sources[1].stopped"))

    def test_replay_waits_for_and_targets_browser_session(self):
        self.open_chat()
        self.page.locator('.msg-play').click()
        self.page.wait_for_function('boundary.sockets.length === 1 && boundary.sockets[0].readyState === 1')
        self.assertFalse(any(path == '/replay' for path, _, _ in self.requests))
        self.page.evaluate("boundary.sockets[0].emit({type: 'session', satellite_id: 'browser-1'})")
        self.page.wait_for_function("!document.querySelector('.msg-play').disabled")
        requests = [json.loads(body) for path, _, body in self.requests if path == '/replay']
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]['satellite_id'], 'browser-1')

    def test_logo_respects_reduced_motion_and_hidden_tabs(self):
        self.page.emulate_media(reduced_motion="reduce")
        self.open_chat()
        self.page.wait_for_function("boundary.logoFrames.length > 0")
        self.page.locator(".msg-play").click()
        self.page.wait_for_function("boundary.sockets.length === 1")
        self.page.evaluate("""() => {
          boundary.sockets[0].emit({type: 'tts_start'});
          boundary.sockets[0].emit(new Float32Array(6000).buffer);
        }""")
        self.assertEqual(self.page.evaluate("boundary.logoFrames.at(-1)"), 0)
        self.page.emulate_media(reduced_motion="no-preference")
        self.page.wait_for_function("boundary.logoFrames.at(-1) >= 8")
        self.page.evaluate("""() => {
          Object.defineProperty(document, 'hidden', {configurable: true, value: true});
          document.dispatchEvent(new Event('visibilitychange'));
        }""")
        self.assertEqual(self.page.evaluate("boundary.logoFrames.at(-1)"), 0)
        self.page.evaluate("""() => {
          Object.defineProperty(document, 'hidden', {configurable: true, value: false});
          document.dispatchEvent(new Event('visibilitychange'));
        }""")
        self.assertGreaterEqual(self.page.evaluate("boundary.logoFrames.at(-1)"), 8)

    def test_chat_history_trace_cards_stats_commands_and_reset(self):
        self.open_chat()
        self.page.get_by_role("button", name="Kitchen", exact=True).click()
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
        chat_bodies = [json.loads(body) for path, _, body in self.requests if path == "/chat"]
        self.assertEqual(chat_bodies, [
            {"text": "What is playing?", "ha_area": "kitchen"},
            {"text": "pause Kitchen", "ha_area": "kitchen"},
        ])
        self.assertEqual(self.page.evaluate("boundary.sockets.length"), 0)
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

    def test_thinking_card_is_concise_and_polling_recovers_final_report_without_sse(self):
        self.page.clock.install()
        job = {
            "id": "research-1", "task": "Research Solar League\n\nAdditional user constraints: Official schedules",
            "objective": "Research Solar League", "status": "RUNNING", "stage": "Planning research",
            "job_duration_seconds": 9.8, "acknowledgement_seconds": 0.003,
            "progress": {}, "request_metrics": [{"input_tokens": 2252, "output_reservation": 2048}],
        }
        self.status["thinking_job"] = job
        self.open_chat()
        self.page.wait_for_function("document.querySelector('#thinking-job-stage').textContent === 'Planning research'")
        self.assertEqual(self.page.locator("#thinking-job-task").inner_text(), "Research Solar League")
        self.assertEqual(self.page.locator("#thinking-job-outcome").inner_text(), "Elapsed: 0:09")
        self.assertFalse(self.page.locator("#thinking-job-details").evaluate("el => el.open"))
        visible = self.page.locator("#thinking-job-card").inner_text()
        self.assertNotIn("reserved", visible)
        self.assertNotIn("0 processed", visible)
        self.assertNotIn("Additional user constraints", visible)
        self.page.clock.run_for(1100)
        self.assertEqual(self.page.locator("#thinking-job-outcome").inner_text(), "Elapsed: 0:10")
        self.page.locator("#thinking-job-details summary").click()
        self.assertIn("2252 in / 2048 reserved out", self.page.locator("#thinking-job-diagnostics").inner_text())

        self.status["thinking_job"] = {
            **job, "stage": "Writing chapters", "job_duration_seconds": 15,
            "progress": {"processed_unique_documents": 3, "fetched_documents": 4, "saved_sections": 1},
        }
        self.page.clock.run_for(5000)
        self.page.wait_for_function("document.querySelector('#thinking-job-stage').textContent === 'Writing chapters'")
        self.assertIn("3 sources reviewed · 1 section saved", self.page.locator("#thinking-job-outcome").inner_text())
        self.assertTrue(self.page.locator("#thinking-job-details").evaluate("el => el.open"))

        final = {**self.status["thinking_job"], "status": "READY", "stage": "Report ready",
                 "job_duration_seconds": 20, "partial_artifact_link": "/reports/Fulloch/example.md"}
        self.page.route("**/thinking/research-1", lambda route: route.fulfill(json=final))
        self.status["thinking_job"] = None
        self.page.clock.run_for(5000)
        self.page.wait_for_function("document.querySelector('#thinking-job-stage').textContent === 'Report ready'")
        self.assertEqual(self.page.locator("#thinking-job-report").get_attribute("href"), "/reports/Fulloch/example.md")
        self.assertTrue(self.page.locator("#thinking-job-cancel").is_visible())
        self.assertEqual(self.page.locator(".thinking-orbit").evaluate("el => getComputedStyle(el).animationName"), "none")
        self.page.clock.run_for(10000)
        self.assertEqual(self.page.locator("#thinking-job-outcome").inner_text(),
                         "Duration: 0:20 · 3 sources reviewed · 1 section saved")
        self.page.evaluate("window.dispatchEvent(new Event('pagehide'))")
        self.assertEqual(self.page.evaluate("boundary.intervals.size"), 0)

    def test_thinking_sse_tracks_job_identity_and_keeps_failure_diagnostics_in_details(self):
        self.page.clock.install()
        self.status["thinking_job"] = {
            "id": "old", "task": "Old request", "status": "RUNNING", "stage": "Writing chapters",
            "job_duration_seconds": 100, "progress": {"processed_unique_documents": 8},
            "partial_artifact_link": "/reports/old.md",
        }
        self.open_chat()
        self.page.wait_for_function("document.querySelector('#thinking-job-task').textContent === 'Old request'")
        self.page.locator("#thinking-job-details summary").click()
        job = {"job_id": "new", "task": "New request", "status": "RUNNING", "stage": "Planning research",
               "job_duration_seconds": 2, "progress": {}}
        self.status["thinking_job"] = {**job, "id": "new"}
        with self.page.expect_response("**/status"):
            self.emit({"role": "thinking", **job})
        self.page.wait_for_timeout(30)
        self.page.wait_for_function("document.querySelector('#thinking-job-task').textContent === 'New request'")
        self.assertEqual(self.page.locator("#thinking-job-outcome").inner_text(), "Elapsed: 0:02")
        self.assertFalse(self.page.locator("#thinking-job-details").evaluate("el => el.open"))
        self.assertFalse(self.page.locator("#thinking-job-report").is_visible())
        self.page.clock.run_for(2100)
        self.assertRegex(self.page.locator("#thinking-job-outcome").inner_text(), r"Elapsed: 0:0[34]")

        self.status["thinking_job"] = None
        self.emit({"role": "thinking", "job_id": "new", "status": "FAILED",
                   "stage": "Research failed during planning", "job_duration_seconds": 4,
                   "error": "ReportSynthesisError: internal details"})
        self.assertEqual(self.page.locator("#thinking-job-stage").inner_text(), "Couldn’t complete the report")
        self.assertNotIn("ReportSynthesisError", self.page.locator("#thinking-job-card").inner_text())
        self.page.locator("#thinking-job-details summary").click()
        self.assertIn("ReportSynthesisError", self.page.locator("#thinking-job-diagnostics").inner_text())
        self.page.clock.run_for(5000)
        self.assertEqual(self.page.locator("#thinking-job-outcome").inner_text(), "Duration: 0:04")

    def test_thinking_stale_status_response_cannot_overwrite_newer_sse_progress(self):
        self.page.clock.install()
        initial = {"id": "research-1", "task": "Research Solar League", "status": "RUNNING", "stage": "Planning research",
                   "job_duration_seconds": 1, "progress": {}}
        self.status["thinking_job"] = initial
        self.open_chat()
        self.page.wait_for_function("document.querySelector('#thinking-job-stage').textContent === 'Planning research'")
        held = []
        self.page.route("**/status", lambda route: held.append(route))
        self.page.clock.run_for(5000)
        self.page.wait_for_timeout(30)
        older_requests = list(held)
        self.assertGreaterEqual(len(older_requests), 1)
        newer = {**initial, "stage": "Reading sources", "job_duration_seconds": 6,
                 "progress": {"processed_unique_documents": 2}}
        self.emit({"role": "thinking", **newer, "job_id": newer["id"]})
        self.page.wait_for_timeout(30)
        self.assertGreater(len(held), len(older_requests))
        for route in held[len(older_requests):]:
            route.fulfill(json={**self.status, "thinking_job": newer})
        self.page.wait_for_function("document.querySelector('#thinking-job-outcome').textContent.includes('2 sources reviewed')")
        for route in older_requests:
            route.fulfill(json={**self.status, "thinking_job": initial})
        self.page.wait_for_timeout(30)
        self.assertEqual(self.page.locator("#thinking-job-stage").inner_text(), "Reading sources")
        self.assertIn("2 sources reviewed", self.page.locator("#thinking-job-outcome").inner_text())

    def test_typed_room_context_uses_persisted_choice_and_clears_when_skipped(self):
        self.page.add_init_script("localStorage.setItem('sat_ha_area', 'kitchen'); "
                                  "localStorage.setItem('sat_ha_area_decided', '1');")
        self.open_chat()
        self.page.locator("#input").fill("turn off lights")
        self.page.locator("#input").press("Enter")
        self.page.wait_for_timeout(30)
        bodies = [json.loads(body) for path, _, body in self.requests if path == "/chat"]
        self.assertEqual(bodies, [{"text": "turn off lights", "ha_area": "kitchen"}])
        self.emit({"role": "stopped"})
        self.page.locator("#sat-area-pill").click()
        self.page.get_by_role("button", name="Skip", exact=True).click()
        self.page.locator("#input").fill("turn on lights")
        self.page.locator("#input").press("Enter")
        self.page.wait_for_timeout(30)
        bodies = [json.loads(body) for path, _, body in self.requests if path == "/chat"]
        self.assertEqual(bodies[-1], {"text": "turn on lights", "ha_area": None})
        self.assertEqual(self.page.evaluate("boundary.sockets.length"), 0)

    def test_invalid_typed_room_shows_error_and_releases_send_button(self):
        self.page.route("**/chat", lambda route: route.fulfill(
            status=400, content_type="application/json",
            body=json.dumps({"detail": "Please select the room again; no command was sent."}),
        ))
        self.open_chat()
        self.page.locator("#input").fill("turn off lights")
        self.page.locator("#input").press("Enter")
        self.page.get_by_text("Please select the room again; no command was sent.", exact=True).wait_for()
        self.assertEqual(self.page.locator("#send").get_attribute("title"), "Send")

    def test_documents_tab_lists_markdown_documents_in_new_tabs(self):
        self.open_chat()
        self.page.get_by_role("tab", name="Documents", exact=True).click()
        self.assertFalse(self.page.locator('.documents-folder').evaluate('(el) => el.open'))
        self.page.locator('.documents-folder summary').click()
        document = self.page.locator(".document-file")
        self.assertEqual(document.inner_text(), "example report")
        self.assertEqual(document.get_attribute("href"), "/documents/Fulloch/example-report.md")
        self.assertEqual(document.get_attribute("target"), "_blank")

    def test_document_semantic_search_results_clear_errors_and_stale_responses(self):
        self.documents = {"documents": [{"path": "Folder/a note.md", "title": "a note"}]}

        def search_route(route):
            query = route.request.post_data_json["query"]
            if query == "failure":
                return route.fulfill(status=503, json={"detail": "unavailable"})
            results = [] if query == "empty" else [{
                "path": "Folder/a note.md", "title": f"Match {query}",
                "excerpt": "<img src=x onerror=alert(1)> relevant excerpt",
            }]
            return route.fulfill(json={"results": results})

        self.page.route("**/api/documents/search", search_route)
        self.open_chat()
        self.page.get_by_role("tab", name="Documents", exact=True).click()
        self.page.locator(".documents-folder summary").wait_for()
        search = self.page.get_by_role("searchbox", name="Search documents by meaning")
        search.fill("herbs")
        search.press("Enter")
        result = self.page.locator(".document-search-result")
        self.assertEqual(result.locator("a").inner_text(), "Match herbs")
        self.assertEqual(result.locator("a").get_attribute("href"), "/documents/Folder/a%20note.md")
        self.assertEqual(result.locator("a").get_attribute("target"), "_blank")
        self.assertIn("<img", result.locator(".document-search-excerpt").inner_text())
        self.assertEqual(result.locator("img").count(), 0)
        self.assertTrue(self.page.locator("#documents-sort").is_disabled())
        self.page.get_by_role("button", name="Clear", exact=True).click()
        self.assertFalse(self.page.locator("#documents-sort").is_disabled())
        self.page.locator(".documents-folder summary").click()
        self.assertEqual(self.page.locator(".document-file").inner_text(), "a note")

        search.fill("empty")
        search.press("Enter")
        self.page.get_by_text("No matching documents found.").wait_for()
        search.fill("failure")
        search.press("Enter")
        self.page.get_by_text("Could not search documents. Please try again.").wait_for()

        self.page.evaluate("""() => {
          const original = window.fetch;
          window.fetch = async (...args) => {
            if (args[0] === '/api/documents/search' && JSON.parse(args[1].body).query === 'old') {
              await new Promise(resolve => { window.releaseOldDocumentSearch = resolve; });
              const response = await original(...args);
              window.oldDocumentFinished = true;
              return response;
            }
            return original(...args);
          };
        }""")
        search.fill("old")
        search.press("Enter")
        self.page.wait_for_function("typeof window.releaseOldDocumentSearch === 'function'")
        search.fill("new")
        search.press("Enter")
        self.page.get_by_text("Match new", exact=True).wait_for()
        self.page.evaluate("window.releaseOldDocumentSearch()")
        self.page.wait_for_function("window.oldDocumentFinished === true")
        self.page.wait_for_function("document.querySelector('.document-file')?.textContent === 'Match new'")
        self.assertEqual(self.page.locator(".document-file").inner_text(), "Match new")

    def test_report_card_keeps_mobile_navigation_reachable(self):
        self.page.set_viewport_size({"width": 375, "height": 667})
        self.open_chat()
        self.emit({
            "role": "assistant", "source": "text", "ts": 2, "content": "Report ready",
            "artifact": {
                "type": "generated_report", "title": "A detailed report",
                "summary": "A concise summary of the generated report.",
                "created_at": 1_700_000_000,
                "report_url": "/reports/fulloch-reports/2026-09-24-deadbeef",
            },
        })
        metrics = self.page.evaluate("""() => ({
          documentWidth: document.documentElement.scrollWidth,
          viewportWidth: document.documentElement.clientWidth,
          chatWidth: document.querySelector('#chat').clientWidth,
          cardWidth: document.querySelector('.generated-report-card').getBoundingClientRect().width,
          settingsRight: document.querySelector('#settings-btn').getBoundingClientRect().right,
          tabsScrollable: document.querySelector('nav.tabs').scrollWidth > document.querySelector('nav.tabs').clientWidth,
        })""")
        self.assertLessEqual(metrics["documentWidth"], metrics["viewportWidth"])
        self.assertLessEqual(metrics["cardWidth"], metrics["chatWidth"])
        self.assertLessEqual(metrics["settingsRight"], metrics["viewportWidth"])
        self.assertTrue(metrics["tabsScrollable"])
        self.page.get_by_role("tab", name="Satellites", exact=True).click()
        self.assertFalse(self.page.locator("#satellites").is_hidden())

    def test_document_sorting_and_folder_pagination(self):
        self.documents = {"documents": [
            {"path": f"Reports/note-{i}.md", "title": f"Note {i}", "modified_at": i}
            for i in range(30, 0, -1)
        ]}
        self.open_chat()
        self.page.get_by_role("tab", name="Documents", exact=True).click()
        self.page.locator('.documents-folder summary').click()
        links = self.page.locator('.document-file')
        self.assertEqual(links.first.inner_text(), 'Note 1')
        self.assertEqual(links.count(), 25)
        self.page.get_by_role('button', name='Next', exact=True).click()
        self.assertEqual(links.first.inner_text(), 'Note 26')
        self.assertEqual(links.count(), 5)
        self.assertTrue(self.page.get_by_role('button', name='Next', exact=True).is_disabled())
        self.page.get_by_role('button', name='Previous', exact=True).click()
        self.assertEqual(links.first.inner_text(), 'Note 1')
        self.page.locator('#documents-sort').select_option('recent')
        self.page.locator('.documents-folder summary').click()
        self.assertEqual(links.first.inner_text(), 'Note 30')
        self.page.get_by_role('button', name='Next', exact=True).click()
        self.assertEqual(links.first.inner_text(), 'Note 5')

    def test_document_edit_save_and_retry(self):
        from server.routes_chat import _document_html

        saved = []

        def document_route(route):
            if route.request.method == 'GET':
                return route.fulfill(body=_document_html('Example', '# Original'), content_type='text/html')
            saved.append(route.request.post_data_json['content'])
            if len(saved) == 1:
                return route.fulfill(status=500)
            return route.fulfill(json={'html': '<h1>Edited</h1>'})

        self.page.route('**/documents/example.md', document_route)
        self.page.goto('http://localhost/documents/example.md')
        self.page.get_by_role('button', name='Edit document').click()
        editor = self.page.get_by_role('textbox', name='Markdown document')
        self.assertEqual(editor.input_value(), '# Original')
        editor.fill('# Edited')
        self.page.get_by_role('button', name='Save document').click()
        self.page.locator('#document-error').wait_for(state='visible')
        self.assertEqual(editor.input_value(), '# Edited')
        self.page.get_by_role('button', name='Save document').click()
        self.page.locator('#document-content').wait_for(state='visible')
        self.assertEqual(self.page.locator('#document-content h1').inner_text(), 'Edited')
        self.assertEqual(saved, ['# Edited', '# Edited'])
        self.page.get_by_role('button', name='Edit document').click()
        self.assertEqual(editor.input_value(), '# Edited')

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

    def test_wizard_defaults_navigation_install_progress_and_public_retry(self):
        self.status["phase"] = "NEEDS_SETUP"
        self.page.goto("http://localhost/setup")
        self.assertEqual(self.page.locator('#tier-select').input_value(), 'lite')
        self.assertEqual(self.page.locator('#tier-select option').count(), 1)
        self.page.locator('#wizard-wakeword').select_option('alexa')
        self.assertIn('not gating', self.page.locator('#wakeword-warning').inner_text())
        self.page.locator("#next1").click()
        self.page.locator('#ha-url').fill('http://ha:8123')
        self.page.locator('#ha-next').click()
        self.page.locator('#search-back').click()
        self.assertEqual(self.page.locator('#ha-url').input_value(), 'http://ha:8123')
        self.page.locator('#ha-next').click()
        self.progress.update(state="error", error="Access required", needs_hf_token=True)
        self.page.locator("#get-started").click()
        self.page.locator("#hf-token").fill("hf_test")
        self.page.get_by_role("button", name="Save token and retry").click()
        self.page.wait_for_timeout(50)
        self.assertTrue(any(path == "/setup/retry-download" for path, _, _ in self.requests))
        models = [json.loads(body)["models"] for path, _, body in self.requests if path == "/setup/models"]
        self.assertEqual(models[0]["llm"]["backend"], "laya")
        self.assertTrue(models[0]['wakeword']['model'].endswith('alexa_v0.1.onnx'))
        configs = [json.loads(body)["updates"] for path, method, body in self.requests
                   if path == "/config" and method == "PUT"]
        self.assertEqual(configs[0]["general.voice_clone"], "af_heart")
        self.assertEqual(configs[0]['general.wakeword'], 'alexa')
        self.progress.update(state="done")
        self.status.update(phase="READY")
        self.page.get_by_role("button", name="Retry download", exact=True).click()
        self.page.wait_for_url('http://localhost/')

    def test_settings_cards_fields_and_model_save(self):
        self.schema["models"] = next(t['models'] for t in self.schema['tier_presets'] if t['id'] == 'lite')
        self.page.goto("http://localhost/setup")
        self.page.locator("#save-cfg").wait_for()
        self.page.locator("details.settings-card").evaluate_all("nodes => nodes.forEach(n => n.open = true)")
        self.page.locator("#sm-ai").select_option("custom")
        self.page.locator("#sm-ai-address").fill("localhost:11434")
        self.page.locator("#sm-api-model").fill("test-model")
        self.page.locator("#save-cfg").click()
        self.page.wait_for_timeout(50)
        models = [json.loads(body)["models"] for path, _, body in self.requests if path == "/setup/models"]
        self.assertEqual(models[0]["llm"]["base_url"], "http://localhost:11434/v1")
        self.assertEqual(models[0]["llm"]["model"], "test-model")
        self.page.locator("#voice-sel").select_option("af_bella")
        self.page.locator("#save-cfg").click()
        self.page.wait_for_timeout(50)
        self.assertTrue(any(path == "/config" and 'af_bella' in (body or '') for path, _, body in self.requests))

    def test_wizard_language_selection_prepares_and_persists(self):
        self.status["phase"] = "NEEDS_SETUP"
        self.page.goto("http://localhost/setup")
        picker = self.page.locator("#wizard-intent-languages")
        self.assertIn("English", picker.locator(".language-pill").inner_text())
        picker.locator("summary").click()
        picker.locator('input[value="fr"]').click()
        self.assertEqual(picker.locator(".language-pill").count(), 2)
        self.page.locator("#next1").click()
        self.page.wait_for_function("document.querySelector('#intent-preparation')?.textContent.includes('ready')")
        prepared = [json.loads(body)["languages"] for path, _, body in self.requests if path == "/setup/intent-languages"]
        self.assertEqual(prepared, [["en", "fr"]])
        self.assertFalse(any(path == "/config" and method == "PUT" for path, method, _ in self.requests))
        self.page.locator("#ha-back").click()
        self.assertEqual(picker.locator(".language-pill").count(), 2)
        self.page.locator("#next1").click()
        self.page.locator("#ha-next").click()
        self.page.locator("#get-started").click()
        self.page.wait_for_function("!!document.querySelector('#cancel-startup')")
        configs = [json.loads(body)["updates"] for path, method, body in self.requests if path == "/config" and method == "PUT"]
        self.assertEqual(configs[-1]["general.intent_languages"], ["en", "fr"])

    def test_wizard_tts_language_defaults_and_persists(self):
        self.status['phase'] = 'NEEDS_SETUP'
        self.schema['models'] = {'asr': {'backend': 'moonshine'},
                                 'tts': {'backend': 'pocket-tts-onnx'}, 'llm': {'backend': 'laya'}}
        self.page.goto('http://localhost/setup')
        self.page.locator('#tier-select').wait_for()
        select = self.page.locator('#wizard-tts-language')
        self.assertTrue(select.is_visible())
        self.assertEqual(select.input_value(), 'english')
        picker = self.page.locator('#wizard-intent-languages')
        picker.locator('summary').click()
        picker.locator('input[value="de-CH"]').check()
        picker.locator('input[value="fr"]').check()
        self.assertEqual(select.input_value(), 'german')
        picker.locator('input[value="de-CH"]').uncheck()
        self.assertEqual(select.input_value(), 'french')
        select.select_option('italian')
        picker.locator('input[value="es"]').check()
        self.assertEqual(select.input_value(), 'italian')
        self.page.locator('#next1').click()
        self.page.locator('#ha-back').click()
        self.assertEqual(select.input_value(), 'italian')
        self.page.locator('#next1').click()
        self.page.locator('#ha-next').click()
        self.page.locator('#get-started').click()
        self.page.wait_for_function("!!document.querySelector('#cancel-startup')")
        updates = [json.loads(body)['updates'] for path, method, body in self.requests if path == '/config' and method == 'PUT']
        self.assertEqual(updates[-1]['general.tts_language'], 'italian')

    def test_wizard_tts_language_updates_with_preset(self):
        self.status['phase'] = 'NEEDS_SETUP'
        self.schema['models'] = {'asr': {'backend': 'moonshine'},
                                 'tts': {'backend': 'pocket-tts-onnx'}, 'llm': {'backend': 'laya'}}
        self.page.goto('http://localhost/setup')
        self.page.locator('#tier-select').wait_for()
        self.assertTrue(self.page.locator('#wizard-tts-language').is_visible())
        self.page.locator('#tier-select').select_option('lite')
        self.assertTrue(self.page.locator('#wizard-tts-language-field').is_hidden())

    def test_wizard_tts_language_hidden_for_unsupported_models(self):
        self.status['phase'] = 'NEEDS_SETUP'
        self.page.goto('http://localhost/setup')
        self.assertTrue(self.page.locator('#wizard-tts-language-field').is_hidden())
        self.page.locator('#next1').click()
        self.page.locator('#ha-next').click()
        self.page.locator('#get-started').click()
        self.page.wait_for_function("!!document.querySelector('#cancel-startup')")
        updates = [json.loads(body)['updates'] for path, method, body in self.requests if path == '/config' and method == 'PUT']
        self.assertNotIn('general.tts_language', updates[-1])

    def test_wizard_language_download_failure_blocks_install_until_retry(self):
        self.status["phase"] = "NEEDS_SETUP"
        self.intent_prepare_error = True
        self.page.goto("http://localhost/setup")
        self.page.locator("#wizard-intent-languages summary").click()
        self.page.locator('#wizard-intent-languages input[value="fr"]').check()
        self.page.locator("#next1").click()
        self.page.wait_for_function("document.querySelector('#intent-preparation')?.textContent.includes('Could not')")
        self.page.locator("#ha-next").click()
        self.page.locator("#get-started").click()
        self.assertIn("retry", self.page.locator("#alert").inner_text())
        self.assertFalse(any(path == "/setup/install" for path, _, _ in self.requests))
        self.intent_prepare_error = False
        self.page.locator("#search-back").click()
        self.page.locator("#ha-back").click()
        self.page.locator("#next1").click()
        self.page.wait_for_function("document.querySelector('#intent-preparation')?.textContent.includes('ready')")

    def test_settings_language_autosave_and_failure_restore(self):
        self.page.goto("http://localhost/setup")
        self.page.locator('[data-section="General"] > summary').click()
        picker = self.page.locator("#settings-intent-languages")
        picker.locator("summary").click()
        picker.locator('input[value="de"]').check()
        self.page.wait_for_function("document.querySelector('#intent-language-status').textContent.includes('saved')")
        configs = [json.loads(body)["updates"] for path, method, body in self.requests if path == "/config" and method == "PUT"]
        self.assertEqual(configs[-1], {"general.intent_languages": ["en", "de"]})
        self.config_error = True
        picker.locator("summary").click()
        picker.locator('input[value="fr"]').click()
        self.page.wait_for_function("document.querySelector('#intent-language-status').textContent.includes('failed')")
        self.assertEqual(picker.locator(".language-pill").count(), 2)
        picker.locator("summary").click()
        self.assertFalse(picker.locator('input[value="fr"]').is_checked())

    def test_settings_can_clear_existing_english_asr_override(self):
        asr = next(field for field in self.schema["fields"] if field["path"] == "general.asr_language")
        asr.update(value="English", set=True)
        self.page.goto("http://localhost/setup")
        self.page.locator('[data-section="General"] > summary').click()
        override = self.page.locator('#cf-general_asr_language')
        self.assertEqual(override.input_value(), "English")
        override.fill("")
        self.page.locator('#save-cfg').click()
        self.page.wait_for_function("document.querySelector('#save-note')?.textContent.includes('Saved')")
        configs = [json.loads(body)["updates"] for path, method, body in self.requests if path == "/config" and method == "PUT"]
        self.assertEqual(configs[-1]["general.asr_language"], "")

    def test_language_selector_requires_at_least_one_language(self):
        self.status["phase"] = "NEEDS_SETUP"
        self.page.goto("http://localhost/setup")
        picker = self.page.locator("#wizard-intent-languages")
        picker.locator("summary").focus()
        self.page.keyboard.press("Enter")
        picker.locator('input[value="en"]').click()
        self.assertTrue(picker.locator('input[value="en"]').is_checked())
        self.assertIn("at least one", picker.locator(".language-validation").inner_text())

    def test_settings_uses_runtime_defaults_not_first_gpu_dropdown_option(self):
        self.schema["variant"] = "gpu"
        self.schema['default_backends'] = {'asr': 'qwen', 'tts': 'qwen', 'llm': 'llama'}
        self.schema["backends"]["asr"] = [
            {"backend": "parakeet", "offerable": True, "display_name": "Parakeet"},
            {"backend": "qwen", "offerable": True, "display_name": "Qwen"},
        ]
        self.schema["backends"]["tts"] = [
            {"backend": "higgs-gguf", "offerable": True, "display_name": "Higgs"},
            {"backend": "qwen", "offerable": True, "display_name": "Qwen"},
        ]
        self.page.goto("http://localhost/setup")

        self.assertEqual(self.page.locator("#sm-asr").input_value(), "qwen")
        self.assertEqual(self.page.locator("#sm-tts").input_value(), "qwen")

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

    def test_note_artifacts_link_to_exact_documents(self):
        self.open_chat()
        self.emit({"role": "assistant", "source": "text", "ts": 2, "content": "Result",
                   "artifact": {"type": "note", "title": "Note", "excerpt": "Text", "path": "folder/My note.md"}})
        self.emit({"role": "assistant", "source": "text", "ts": 3, "content": "Result",
                   "artifact": {"type": "notes_search", "query": "Note", "matches": [
                       {"title": "Report", "excerpt": "Text", "path": "fulloch-reports/report #1.md"},
                       {"title": "Unsafe", "excerpt": "Text", "path": "../other.md"},
                       {"title": "Old", "excerpt": "Text"}]}})
        note = self.page.locator('.note-card a')
        self.assertEqual(note.get_attribute('href'), '/documents/folder/My%20note.md')
        self.assertEqual(note.get_attribute('target'), '_blank')
        result = self.page.locator('.notes-search-match').first
        self.assertEqual(result.get_attribute('href'), '/documents/fulloch-reports/report%20%231.md')
        self.assertEqual(result.get_attribute('rel'), 'noopener noreferrer')
        self.assertEqual(self.page.locator('.notes-search-match:not(a)').count(), 2)

    def test_report_paper_and_web_source_links(self):
        self.open_chat()
        url = '/reports/fulloch-reports/2026-09-11-1234abcd'
        report = {"title": "Report", "summary": "Summary", "created_at": 1, "report_url": url}
        artifacts = [
            {"type": "generated_report", **report},
            {"type": "research_report", **report, "data": {"type": "paper_search", "papers": [
                {"title": "Paper", "url": "https://example.com/paper"},
                {"title": "Unsafe", "url": "javascript:bad()"}]}},
            {"type": "travel_report", "title": "No link", "summary": "Summary", "created_at": 1},
            {"type": "flight_plan", "route": {"origin": "A", "destination": "B"}, "offer": {}, "report_url": url},
            {"type": "web_research", "query": "Topic", "sources": [
                {"url": "https://example.com", "host": "Example", "evidence": "Excerpt"}]},
        ]
        for ts, artifact in enumerate(artifacts, 2):
            self.emit({"role": "assistant", "source": "text", "ts": ts, "content": "Result", "artifact": artifact})
        previews = self.page.locator('.report-preview-link')
        self.assertEqual(previews.count(), 3)
        for preview in previews.all():
            self.assertEqual(preview.get_attribute('href'), url)
            self.assertEqual(preview.get_attribute('target'), '_blank')
        self.assertEqual(self.page.locator('.travel_report .report-preview-link').count(), 0)
        paper = self.page.locator('.research_report a.domain-report-row')
        self.assertEqual(paper.get_attribute('href'), 'https://example.com/paper')
        self.assertEqual(paper.get_attribute('rel'), 'noopener noreferrer')
        self.assertEqual(self.page.locator('.research_report div.domain-report-row').count(), 1)
        source = self.page.locator('.web-source')
        self.assertIn('Show excerpt', source.locator('summary').inner_text())
        source.locator('summary').click()
        self.assertTrue(source.locator('p').is_visible())
        self.assertEqual(source.locator('a').inner_text(), 'Open source')
        self.assertEqual(source.locator('a').get_attribute('target'), '_blank')

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

    def test_settings_voice_preview_and_config_only_controls(self):
        self.page.goto("http://localhost/setup")
        self.page.locator('#save-cfg').wait_for()
        self.page.locator('details.settings-card').evaluate_all('nodes => nodes.forEach(n => n.open = true)')
        self.page.locator('.voice-play').click()
        self.assertFalse(self.page.evaluate('boundary.previews[0].paused'))
        self.page.locator('#voice-sel').select_option('af_bella')
        self.assertTrue(self.page.evaluate('boundary.previews[0].paused'))
        self.assertTrue(self.page.locator('#custom-voice').is_hidden())
        self.assertEqual(self.page.locator('#wakeword-custom-fields, #sm-wakeword-model, #cf-general_wakeword_pattern, #sec-obs-section').count(), 0)
        self.assertEqual(self.page.locator('[data-section="Advanced"]').count(), 0)
        self.assertEqual(self.page.locator('[data-section]').count(), 7)

    def test_cpu_settings_never_offer_gpu_models(self):
        self.page.goto('http://localhost/setup')
        self.page.locator('#save-cfg').wait_for()
        for domain, selector in [('asr', '#sm-asr'), ('tts', '#sm-tts'), ('llm', '#sm-ai')]:
            values = self.page.locator(selector + ' option').evaluate_all('nodes => nodes.map(n => n.value)')
            gpu = [b['backend'] for b in self.schema['backends'][domain] if b['gpu_only']]
            self.assertFalse(set(values) & set(gpu))

    def test_preset_pills_reflect_customisation_and_reapply(self):
        self.schema['models'] = next(t['models'] for t in self.schema['tier_presets'] if t['id'] == 'lite')
        self.page.goto('http://localhost/setup')
        self.page.locator('#save-cfg').wait_for()
        pill = self.page.locator('.preset-pill[data-tier="lite"]')
        self.assertEqual(pill.get_attribute('aria-pressed'), 'true')
        self.assertNotIn('customised', pill.get_attribute('class'))
        self.page.locator('[data-section="General"]').evaluate('node => node.open = true')
        self.page.locator('#cf-notes_path').fill('/custom/notes')
        self.assertIn('customised', pill.get_attribute('class'))
        self.assertIn('Customised from', self.page.locator('#preset-description').inner_text())
        pill.click()
        self.page.wait_for_function("!document.querySelector('#save-cfg').disabled")
        self.assertNotIn('customised', pill.get_attribute('class'))
        self.assertEqual(self.page.locator('#cf-notes_path').input_value(), '/custom/notes')

    def test_popovers_close_on_outside_click_and_escape(self):
        self.page.goto('http://localhost/setup')
        self.page.locator('#save-cfg').wait_for()
        popup = self.page.locator('.preset-heading .info-popover')
        popup.locator('summary').click()
        self.assertTrue(popup.get_attribute('open') is not None)
        self.page.locator('#preset-heading').click()
        self.assertIsNone(popup.get_attribute('open'))
        popup.locator('summary').focus()
        self.page.keyboard.press('Enter')
        self.page.keyboard.press('Escape')
        self.assertIsNone(popup.get_attribute('open'))

    def test_ungated_pocket_pytorch_and_premium_need_no_token(self):
        self.schema['variant'] = 'gpu'
        for backends in self.schema['backends'].values():
            for backend in backends:
                backend['offerable'] = True
        self.schema['models'] = next(t['models'] for t in self.schema['tier_presets'] if t['id'] == 'premium')
        self.page.goto('http://localhost/setup')
        self.page.locator('#save-cfg').wait_for()
        self.assertEqual(self.page.locator('#preset-access .access-notice').count(), 0)
        self.page.locator('[data-section="Models"]').evaluate('node => node.open = true')
        self.page.locator('#sm-tts').select_option('pocket-tts-pytorch')
        self.assertEqual(self.page.locator('#model-access .access-notice').count(), 0)
        self.page.locator('[data-section="Voice"]').evaluate('node => node.open = true')
        self.assertTrue(self.page.locator('#tts-language').is_visible())
        self.page.locator('#tts-language').select_option('french')
        self.page.locator('#sm-tts').select_option('pocket-tts-onnx')
        self.assertEqual(self.page.locator('#model-access .access-notice').count(), 0)
        self.assertEqual(self.page.locator('#tts-language').input_value(), 'french')

    def test_tts_language_selector_tracks_supported_backend(self):
        self.schema['models'] = {'tts': {'backend': 'pocket-tts-onnx'}}
        self.page.goto('http://localhost/setup')
        self.page.locator('#save-cfg').wait_for()
        self.page.locator('[data-section="Voice"]').evaluate('node => node.open = true')
        self.assertTrue(self.page.locator('#tts-language-field').is_visible())
        self.page.locator('#tts-language').select_option('french')
        self.page.locator('[data-section="Models"]').evaluate('node => node.open = true')
        self.page.locator('#sm-tts').select_option('kokoro-onnx')
        self.assertFalse(self.page.locator('#tts-language-field').is_visible())
        self.page.locator('#sm-tts').select_option('pocket-tts-onnx')
        self.assertEqual(self.page.locator('#tts-language').input_value(), 'french')

    def test_wizard_recommends_hardware_classes_and_falls_back_safely(self):
        self.open_chat()
        results = self.page.evaluate('''async (schema) => {
          const {recommendedTierId} = await import('/static/js/setup-models.js');
          schema.variant = 'gpu';
          return [0, 6, 8, 10, 12, 16, 24, 32].map(vram => recommendedTierId(schema, {
            gpu: {available: vram > 0, vram_gb: vram},
            tier_fit: schema.tier_presets.map(t => ({id: t.id, badge: t.vram_class_gb <= vram ? 'ok' : 'warn'}))
          })).concat(recommendedTierId(schema, null));
        }''', self.schema)
        self.assertEqual(results, ['lite', 'lite', 'economy', 'economy', 'premium', 'pro', 'pro', 'max', 'lite'])

    def test_wizard_continue_only_uses_default_acoustic_wakeword(self):
        self.status['phase'] = 'NEEDS_SETUP'
        self.page.goto('http://localhost/setup')
        self.page.locator('#next1').click()
        self.page.locator('#ha-next').click()
        self.page.locator('#get-started').click()
        self.page.locator('#assets').wait_for(state='attached')
        models = [json.loads(body)['models'] for path, _, body in self.requests if path == '/setup/models'][0]
        self.assertEqual(models['llm']['backend'], 'laya')
        self.assertEqual(models['wakeword']['backend'], 'openwakeword')
        self.assertIn('hey_atticus', models['wakeword']['model'])
        self.assertNotIn('verify_asr', models['wakeword'])  # Runtime selects the safe backend default.
        self.assertFalse(any('obsidian' in path for path, _, _ in self.requests))

    def test_home_assistant_token_flow_in_wizard_and_settings(self):
        self.page.route('**/setup/integration-token', lambda route: route.fulfill(
            body=json.dumps({'token': 'new-fulloch-token', 'port': 8766,
                             'enabled': True, 'bind_host': '127.0.0.1'}),
            content_type='application/json'))
        self.status['phase'] = 'NEEDS_SETUP'
        self.page.goto('http://localhost/setup')
        self.page.locator('#next1').click()
        self.assertIn('different from', self.page.locator('.ha-integration-card').inner_text())
        self.page.locator('#ha-create-integration').click()
        self.page.locator('#ha-integration-result').wait_for()
        self.assertEqual(self.page.locator('#ha-integration-token').input_value(), 'new-fulloch-token')
        self.assertIn('8766', self.page.locator('#ha-integration-address').inner_text())
        self.assertIn('only on this machine', self.page.locator('#ha-integration-network').inner_text())
        self.page.locator('#ha-next').click()
        self.status['phase'] = 'READY'
        self.page.goto('http://localhost/setup')
        self.page.locator('#save-cfg').wait_for()
        self.page.locator('[data-section="Home Assistant"]').evaluate('node => node.open = true')
        self.page.locator('#ha-create-integration').click()
        self.page.locator('#ha-integration-result').wait_for()
        self.assertEqual(self.page.locator('#ha-integration-token').input_value(), 'new-fulloch-token')

    def test_wizard_does_not_install_after_preflight_reports_failure(self):
        self.status['phase'] = 'NEEDS_SETUP'
        self.page.route('**/setup/preflight-download', lambda route: route.fulfill(
            body=json.dumps({'ok': False, 'errors': [{'check': 'disk', 'message': 'Not enough space'}]}),
            content_type='application/json'))
        self.page.goto('http://localhost/setup')
        self.page.locator('#next1').click()
        self.page.locator('#ha-next').click()
        self.page.locator('#get-started').click()
        self.page.locator('#alert .alert-banner').wait_for()
        self.assertIn('Not enough space', self.page.locator('#alert').inner_text())
        self.assertFalse(self.page.locator('#get-started').is_disabled())
        self.assertFalse(any(path in ('/setup/models', '/setup/install') for path, _, _ in self.requests))

    def test_settings_api_key_change_preserves_hidden_llm_options(self):
        self.schema['models'] = {
            'asr': {'backend': 'moonshine'}, 'tts': {'backend': 'kokoro-onnx'},
            'llm': {'backend': 'external', 'base_url': 'http://server:8888/v1', 'model': 'custom-model',
                    'connect_timeout': 1.2, 'read_timeout': 321},
        }
        self.page.goto('http://localhost/setup')
        self.page.locator('#save-cfg').wait_for()
        self.page.locator('details.settings-card').evaluate_all('nodes => nodes.forEach(n => n.open = true)')
        self.page.locator('#sm-api-key').fill('new-key')
        self.page.locator('#save-cfg').click()
        self.page.wait_for_timeout(50)
        self.assertFalse(any(path == '/setup/models' for path, _, _ in self.requests))
        credentials = [json.loads(body) for path, _, body in self.requests if path == '/setup/credential']
        self.assertIn({'key': 'llm_api_key', 'value': 'new-key'}, credentials)

    def test_settings_preset_and_tts_changes_keep_voice_compatible(self):
        self.schema['variant'] = 'gpu'
        for backends in self.schema['backends'].values():
            for backend in backends:
                backend['offerable'] = True
        self.schema['models'] = next(t['models'] for t in self.schema['tier_presets'] if t['id'] == 'pro')
        self.page.goto('http://localhost/setup')
        self.page.locator('#save-cfg').wait_for()
        self.page.locator('details.settings-card').evaluate_all('nodes => nodes.forEach(n => n.open = true)')
        self.page.locator('.preset-pill[data-tier="premium"]').click()
        self.page.wait_for_function("!document.querySelector('#save-cfg').disabled")
        self.assertEqual(self.page.locator('#sm-asr').input_value(), 'orukeet')
        self.assertEqual(self.page.locator('#sm-tts').input_value(), 'pocket-tts-pytorch')
        self.assertEqual(self.page.locator('#sm-ai').input_value(), 'ornith')
        self.assertTrue(self.page.locator('#custom-voice').is_hidden())
        self.page.locator('#voice-sel').select_option('__custom__')
        self.assertTrue(self.page.locator('#custom-voice').is_visible())
        self.page.locator('#voice-reference').fill('/custom/reference.wav')
        self.page.locator('#voice-sel').select_option('atticus')
        self.assertTrue(self.page.locator('#custom-voice').is_hidden())
        self.page.locator('#voice-sel').select_option('__custom__')
        self.assertTrue(self.page.locator('#custom-voice').is_visible())
        self.page.locator('.preset-pill[data-tier="economy"]').click()
        self.page.wait_for_function("!document.querySelector('#save-cfg').disabled")
        self.assertTrue(self.page.locator('#custom-voice').is_hidden())
        self.assertEqual(self.page.locator('#voice-sel').input_value(), 'af_heart')
        self.page.locator('#save-cfg').click()
        self.page.wait_for_timeout(50)
        saved = [json.loads(body)['models'] for path, _, body in self.requests if path == '/setup/models'][0]
        self.assertEqual(saved['llm']['local_model'], 'ornith')
        self.assertEqual(saved['asr']['backend'], 'moonshine')

    def test_cpu_custom_local_model_is_rejected_without_writing(self):
        self.page.goto('http://localhost/setup')
        self.page.locator('#save-cfg').wait_for()
        self.page.locator('details.settings-card').evaluate_all('nodes => nodes.forEach(n => n.open = true)')
        self.page.locator('#sm-ai').select_option('custom')
        self.page.locator('#sm-ai-address').fill('/local/model.gguf')
        self.page.locator('#save-cfg').click()
        self.page.wait_for_timeout(50)
        self.assertIn('require the GPU image', self.page.locator('#save-note').inner_text())
        self.assertFalse(any(path == '/setup/models' for path, _, _ in self.requests))

    def test_settings_info_popovers_are_keyboard_accessible_on_mobile(self):
        self.page.set_viewport_size({'width': 375, 'height': 812})
        self.page.goto('http://localhost/setup')
        self.page.locator('#save-cfg').wait_for()
        help_button = self.page.locator('.preset-heading .info-popover > summary').first
        help_button.focus()
        help_button.press('Enter')
        popup = self.page.locator('.preset-heading .info-popover[open] > p')
        self.assertTrue(popup.is_visible())
        bounds = popup.bounding_box()
        self.assertGreaterEqual(bounds['x'], 0)
        self.assertLessEqual(bounds['x'] + bounds['width'], 375)
        help_button.press('Enter')
        self.page.locator('[data-section="Models"]').evaluate('node => node.open = true')
        self.page.locator('#sm-ai').select_option('custom')
        help_button = self.page.locator('#sm-ai-custom label .info-popover > summary').first
        help_button.focus()
        help_button.press('Enter')
        bounds = self.page.locator('#sm-ai-custom .info-popover[open] > p').bounding_box()
        self.assertGreaterEqual(bounds['x'], 0)
        self.assertLessEqual(bounds['x'] + bounds['width'], 375)

    def test_unrelated_settings_preserve_custom_models_and_wakeword(self):
        original = {
            'asr': {'backend': 'moonshine-tiny', 'model': '/custom/asr'},
            'tts': {'backend': 'kokoro-onnx', 'model': '/custom/tts'},
            'llm': {'backend': 'laya', 'confidence_threshold': 0.99},
            'wakeword': {'backend': 'openwakeword', 'model': '/custom/trigger.onnx',
                         'threshold': 0.81, 'smoothing_frames': 4, 'cooldown_ms': 2000, 'verify_asr': True},
        }
        self.schema['models'] = original
        self.page.goto('http://localhost/setup')
        self.page.locator('#save-cfg').wait_for()
        self.page.locator('details.settings-card').evaluate_all('nodes => nodes.forEach(n => n.open = true)')
        self.page.locator('#cf-notes_path').fill('/my/notes')
        self.page.locator('#save-cfg').click()
        self.page.wait_for_timeout(50)
        self.assertFalse(any(path == '/setup/models' for path, _, _ in self.requests))
        self.page.locator('#sm-asr').select_option('moonshine')
        self.page.locator('#save-cfg').click()
        self.page.wait_for_timeout(50)
        saved = [json.loads(body)['models'] for path, _, body in self.requests if path == '/setup/models'][0]
        self.assertEqual(saved['wakeword'], original['wakeword'])
        self.assertEqual(saved['tts'], original['tts'])
        self.assertEqual(saved['llm'], original['llm'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
