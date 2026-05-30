# Fulloch — Feature Plan

An expansion of the ideas captured in `data/notes/fulloch-ideas.md`, grounded in
the current architecture. Each idea is assessed for value, cost, and fit, with a
concrete implementation sketch and a verdict. A prioritised summary and my own
additional ideas follow at the end.

Source ideas:
1. Scheduled reminders / jobs
2. Multi-lingual support
3. Kiwix offline search (swap-in for SearXNG)
4. Vector DB update strategy (when / how is it refreshed?)

---

## Guiding principles (the lens used to rank these)

Fulloch's whole reason for existing is **fully local, privacy-first, conversational
home assistance on a single 16GB GPU**. Every feature is judged against:

- **Does it stay local?** Anything that phones home is a non-starter.
- **Does it fit the VRAM budget?** The pipeline (ASR + TTS + 9B SLM + BGE) is already
  GPU-resident. New models compete for the same 16GB.
- **Does it respect the half-duplex / barge-in audio model?** The mic is muted during
  TTS (half-duplex) or guarded by self-echo suppression (barge-in). Anything that
  produces audio has to cooperate with `_turn_lock`, the `transcribing` flag, and the
  per-turn `TtsSession`.
- **Does it earn its complexity?** This is a personal assistant, not a platform. A
  feature that serves a real daily use case beats a technically-interesting one.

---

## 1. Scheduled reminders / jobs

**What it is:** "Remind me to take the bins out at 7pm", "every weekday at 8am tell me
the weather", "in 20 minutes ask me if the oven's off". A time- or interval-triggered
job that fires a spoken (or note-written) action without the user initiating a turn.

### Positives
- **High everyday value.** This is the single most-requested capability of any voice
  assistant after timers and weather. It's the natural next step from the existing
  timer/alarm support.
- **Strong fit with existing memory.** Fulloch already persists facts and notes; a
  reminder is just a persisted intent with a fire time. It reinforces the "builds a
  picture of you over time" positioning.
- **Partial infrastructure already exists.** `audio/beep_manager.py` plays alarm WAVs,
  and `time_tools.py` already manages `threading.Timer` countdowns. The dispatch
  primitive (a timer that runs a callback) is proven.

### Negatives / challenges
- **Proactive speech is the hard part, not the scheduling.** Today *all* TTS happens
  inside a user-initiated turn that owns `_turn_lock`, mutes the mic, and runs a
  `TtsSession`. A reminder firing while the system is idle must replicate that dance:
  take the lock, set `transcribing = False`, play through `speak_stream`/`play_chunks`,
  restore mic state — and do it without colliding with a turn that's already running or
  a barge-in in flight. This is genuinely fiddly concurrency work.
- **Persistence across restart.** `threading.Timer` is in-memory and dies on restart
  (the existing timers already have this limitation). Reminders *must* survive restarts
  or they're useless, so they need an on-disk store plus a reconciliation pass on boot
  (fire-now anything overdue? skip? ask?).
- **Recurring schedules need real cron semantics.** "Every weekday at 8" means parsing
  RRULE-ish recurrence, timezone/DST correctness (the container already mounts
  `/etc/localtime`), and catch-up policy after downtime.
- **Interrupting the user is socially expensive.** A reminder that talks over a
  conversation, or at 3am, is worse than no reminder. Needs quiet-hours and
  "defer if a turn is active" logic.

### Implementation approach
Two viable paths:

**(A) In-repo scheduler (recommended for control).**
- New `tools/scheduler.py` registering `set_reminder`, `list_reminders`, `cancel_reminder`.
- Persist jobs as a small JSON/markdown store under `data/` (mirrors how notes/facts
  persist). Each job: `{id, fire_at | rrule, action, payload, created, last_fired}`.
- A single long-lived **scheduler thread** started from `Assistant.run()` that sleeps
  until the next due job (heap/`sched`-style), then calls a new
  `Assistant.speak_proactive(text)` method.
- `speak_proactive()` is the critical new primitive: acquire `_turn_lock`
  (non-blocking; if a turn is active, queue the reminder and retry after it ends),
  mute the mic, play a short chime + the reminder via the existing `play_chunks` /
  `speak_stream` path with a fresh `TtsSession`, then restore. This is the ~80% of the
  effort.
- Quiet-hours + "defer while turn active" guard live here.

**(B) Delegate to Home Assistant.**
- For users who run HA, HA already has robust scheduling (automations, `todo`,
  `alarm`). A `create_ha_reminder` tool could push a scheduled automation and let HA
  fire a webhook back to Fulloch when due.
- **Pro:** offloads the cron correctness and persistence. **Con:** only works for HA
  users (reminders should arguably work standalone), and the inbound webhook still
  needs the same `speak_proactive()` primitive — so it doesn't actually save the hard
  part.

Build **(A)**; the `speak_proactive()` primitive it forces is reusable for every other
proactive feature (morning briefing, timer-done announcements, alerts).

### Use cases
- "Remind me to call the dentist tomorrow at 9." ✅ core use case
- "Every morning at 7 tell me today's calendar and weather." ✅ (a "briefing", composes
  existing calendar + weather tools)
- "In 10 minutes remind me to check the laundry." ✅ (a soft timer with a spoken payload
  rather than a beep)

### Verdict: **BUILD — highest priority.**
It's the most-wanted capability, it deepens the assistant's core identity, and the hard
piece (`speak_proactive`) unlocks a whole class of follow-on features. Scope the v1 to
**one-shot and simple daily/weekly recurrences with quiet-hours**; defer full RRULE.

---

## 2. Multi-lingual support

**What it is:** Understand and respond in languages other than English — either a fixed
configured language, or auto-detect per utterance.

### Positives
- **The model stack already supports it.** Qwen3-ASR, Qwen3-TTS, and Qwen3.5-9B are all
  multilingual. There's no new model to load and no VRAM cost — the capability is
  latent in what's already running.
- **Broadens the user base** and is a genuine accessibility win for multilingual
  households.

### Negatives / challenges
- **The plumbing around the models is English-baked**, and that's where all the work is:
  - **Wakeword.** `_build_wakeword_pattern` and the shipped `wakeword_pattern` are
    English-phoneme tolerant (`hey/hay/hi`, `s↔z`). A non-English wakeword needs its own
    tolerant pattern.
  - **Prompts.** Every prompt in `utils/prompts.py` is English. The SLM will largely
    follow the user's language, but the system instructions, tool descriptions, and
    sentinel strings (`User question:`, `Thinking question:`, etc.) are English and
    drive control flow.
  - **Self-echo suppression depends on English word-spelling.** `time_tools.py`
    deliberately spells numbers/dates as English words ("twenty twenty-six") so the
    ASR round-trips the assistant's own voice to matching text and the self-echo check
    fires. This entire mechanism is English-specific and would need a per-language
    equivalent — otherwise barge-in self-triggers.
  - **Regex fast-path** (`utils/intent_catch.py`) matches English phrasings ("play X",
    "stop", "think about X"). In another language these silently fall through to the SLM
    (slower, but functional).
- **Mixed-language self-echo and TTS voice/accent mismatch** add polish problems.
- **Testing burden** multiplies per supported language, with no native-speaker QA on a
  solo project.

### Implementation approach
- Add `general.language: "en"` (default). Thread it into ASR transcribe hints, TTS, and
  a localized prompt set (`utils/prompts/<lang>.py` or a translation layer).
- Provide per-language wakeword patterns and a localized number/date speller for the
  self-echo path — or, simpler, **make self-echo suppression language-agnostic** by
  switching from word-form substring matching to a fuzzy/normalized comparison
  (Levenshtein on token sequences), which removes the English-spelling dependency
  entirely. *(This refactor is independently valuable — see additional ideas.)*
- Auto-detect is tempting but risky given the self-echo coupling; start with a single
  **configured** language.

### Use cases
- A household that speaks Spanish/Mandarin/etc. wants the assistant in their language.
  ✅ real, but niche for the current solo-dev audience.
- Code-switching mid-sentence. ❌ explicitly out of scope for v1.

### Verdict: **DEFER (medium-low priority).**
The capability is "free" at the model layer but the surrounding plumbing — especially
the English-coupled self-echo mechanism — makes a *correct* implementation deceptively
large, and the audience that needs it is small relative to reminders. **However**, the
self-echo-decoupling refactor it motivates is worth doing on its own merits, and once
that's done, basic configured-language support becomes much cheaper. Park this behind
that refactor.

---

## 3. Kiwix offline search (swap-in for SearXNG)

**What it is:** Replace (or sit alongside) the SearXNG web search with
[Kiwix](https://kiwix.org) serving offline ZIM archives (offline Wikipedia, Wiktionary,
Stack Exchange, etc.) for a *truly* network-free deployment.

### Positives
- **Completes the "fully local" promise.** SearXNG still reaches out to the public
  internet; Kiwix is genuinely offline. For air-gapped, privacy-maximalist, or
  poor-connectivity setups this is the real deal.
- **Clean swap surface.** `tools/search_web.py` is beautifully isolated: one tool
  (`external_information`), gated by the `search` config key, returning a
  `User question:` sentinel payload that the orchestrator's inline summariser already
  knows how to compress. A Kiwix tool just has to produce the same sentinel shape.
- **`kiwix-serve` is a simple HTTP API** and fits the existing Docker-compose pattern
  (add a service, like SearXNG).
- **Reliable and fast** — no engine timeouts, no rate limits, no bot-detection (the logs
  already show Brave engine timeouts and X-Forwarded-For warnings from SearXNG).

### Negatives / challenges
- **Static snapshots — no current events.** This is the crux. The *stated purpose* of
  `external_information` is explicitly *time-sensitive* queries: "news, scores, prices,
  today's headlines." Kiwix can't answer any of those. So it's **not a true
  replacement** — it's a complement that serves a *different* query class (reference
  knowledge the SLM might already know anyway).
- **Disk cost.** A full offline-Wikipedia ZIM is ~100GB; the "no images / top articles"
  variants are smaller (a few GB) but still meaningful next to the ~15GB model cache.
- **Overlap with the SLM.** Qwen3.5-9B already carries broad world knowledge. Reference
  lookups Kiwix would serve are often answerable from the model directly, which weakens
  the marginal value (the agent prompt already steers general-knowledge queries *away*
  from web search toward the model).
- **Ranking/snippet quality.** Kiwix full-text search is decent but not tuned like a
  search engine; snippet extraction differs from the HTML/JSON path in `search_web.py`.

### Implementation approach
- New `tools/kiwix_search.py` registering `external_information` (or a distinct
  `offline_reference` alias) — reuse the exact `User question:` sentinel envelope so the
  summariser and `should_replan` work unchanged.
- Add a `kiwix:` config block (`kiwix_url`, ZIM selection); add `kiwix-serve` to a
  compose file with a volume for ZIM files.
- **Make it a mode, not a hard swap:** `search.backend: searxng | kiwix | both`. With
  `both`, prefer Kiwix for reference-style queries and SearXNG for time-sensitive ones
  (the agent prompt already distinguishes these).

### Use cases
- Air-gapped / off-grid / RV / boat deployments. ✅ compelling but niche.
- "What's the capital of X / how does Y work" with no internet. ✅ but often
  SLM-answerable already.
- "Today's news / scores / prices." ❌ Kiwix cannot serve these.

### Verdict: **BUILD as an optional backend — medium priority.**
Low implementation risk thanks to the clean tool boundary, and it meaningfully advances
the headline "fully local" claim. But frame it correctly: an **optional offline
reference backend alongside SearXNG**, not a replacement. Ship it config-gated and off
by default. Do it *after* reminders.

---

## 4. Vector DB update strategy

**What it is:** The original note asks "When is the vectordb updated? Schedule job?
Monitor changes?" — i.e. how/when does semantic note search stay in sync with the notes
folder.

### Current state (mostly already solved — see `tools/notes_index.py`)
- **On write:** `index_file()` runs from the note write/append hooks, synchronously
  drop-and-rebuilding just that file's chunks. The index is always consistent with what
  the assistant itself writes.
- **On read:** `search()` calls `scan()` first, which walks the folder and re-embeds any
  file whose mtime changed since last seen (and drops vanished files). So **external
  edits** (e.g. you edit a note in Obsidian) *are* picked up — on the next query.
- **Persistence:** `data/notes_index.npy` + JSON sidecar, restored on first use and
  warmed at startup via `warm_index()`.

So the core question is largely **already answered**: it's updated on write
(synchronous) and reconciled-by-mtime on every search.

### Remaining gaps worth closing
- **External edits aren't indexed until the *next* query.** If you edit a vault on your
  phone, the first semantic search after that pays the re-embed cost inline (the
  "Known Gaps" note in `CLAUDE.md` flags this). For a few-hundred-note store it's
  sub-second; for a large vault it could add latency to that one query.
- **No proactive/background indexing.** There's no watcher; staleness is only resolved
  lazily.
- **Facts block is startup-only.** Separately noted in `CLAUDE.md`: `recall_facts()`
  builds the "Known facts about the user" prompt block once at boot, so facts added
  mid-session don't enter the system prompt until restart.

### Implementation approach
- **Background scan thread** (cheap, high value): a low-frequency timer (e.g. every
  few minutes, or coalesced after the proactive scheduler from idea #1 exists) that
  calls `scan()` off the hot path, so external edits are indexed before the user asks.
  Guard with the existing `_lock`.
- **Optional filesystem watch** (`watchdog`) for instant re-index on change — nicer, but
  adds a dependency and cross-platform quirks (especially over network/synced vault
  mounts where inotify is unreliable). The polling thread is the pragmatic choice.
- **Mid-session facts refresh:** rebuild the facts block when `remember_fact` runs, or
  on the same background tick, rather than only at startup.

### Use cases
- Heavy Obsidian user editing notes outside Fulloch and expecting search to "just know."
  ✅ this is the real driver.

### Verdict: **PARTIAL — low priority polish.**
The original worry is mostly already handled. The genuine improvement is a **lightweight
background scan + mid-session facts refresh**, which pairs naturally with the scheduler
thread from idea #1 (reuse the same timer). Small, safe, do it opportunistically once
the scheduler exists.

---

## Prioritised summary

| # | Idea | Value | Effort | Risk | Verdict | Priority |
|---|------|-------|--------|------|---------|----------|
| 1 | Scheduled reminders / jobs | High | High | Med (concurrency) | **Build** | **1 — do first** |
| 3 | Kiwix offline backend | Med | Low–Med | Low | **Build (optional, off by default)** | **2** |
| 4 | Vector DB background refresh | Low–Med | Low | Low | **Partial (polish)** | **3 — ride on #1's thread** |
| 2 | Multi-lingual | Med | High | Med–High | **Defer** behind self-echo refactor | **4** |

**Rationale for the ordering:**
- **#1 first** because it's the highest user value *and* it forces the `speak_proactive()`
  primitive that several future features depend on.
- **#3 second** because it's low-risk (clean tool boundary), advances the core "fully
  local" pitch, and is independent of everything else.
- **#4 third** because it's mostly done; the remaining polish piggybacks cheaply on #1's
  background thread.
- **#2 last** because the model-level capability is free but the English-coupled
  plumbing (especially self-echo) makes a correct version expensive, and the audience is
  comparatively small.

---

## Additional ideas & my own thoughts

A few directions that fall out of reading the codebase, roughly in the order I'd reach
for them:

1. **Decouple self-echo suppression from English word-spelling (enabling refactor).**
   Today `time_tools.py` jumps through hoops spelling numbers as words purely so the
   ASR re-transcription matches `_last_spoken_text` for substring-based self-echo
   detection. Replacing that with a normalized fuzzy comparison (token-level
   Levenshtein / ratio threshold) would (a) delete the special-casing, (b) make
   self-echo more robust to ASR variation generally, and (c) unblock multi-lingual.
   **This is the highest-leverage piece of plumbing on the list** even though it isn't a
   user-facing feature — it makes idea #2 tractable and simplifies existing code.

2. **Proactive morning/evening briefing.** Once `speak_proactive()` exists (#1), a
   composed "good morning" — today's calendar + weather + any due reminders — is almost
   free and feels premium. It's just the scheduler firing a multi-tool turn.

3. **Conversation/turn history persistence + recap.** `_history` is in-memory and
   cleared after 90s. A lightweight rolling transcript on disk would enable "what did we
   talk about earlier?" and post-hoc note creation ("save what we just discussed"),
   which fits the notes-centric identity well.

4. **TTS response caching for fixed phrases.** Acks, stalls, and the greeting are already
   pre-rendered. Extending a small LRU cache to other frequent fixed responses
   ("Done.", "I couldn't find anything") would shave perceptible latency for free.

5. **Wake-time "presence" / push-to-talk via dashboard.** The dashboard already streams
   turns; a "talk" button that opens the mic on demand would help noisy environments
   where the always-on wakeword struggles, without adding a wakeword model.

6. **Local TTS for arbitrary long-form readout (notes/articles).** "Read me my note on
   X" / "read me that article" — pairs with Kiwix (#3) and notes. The TTS worker already
   streams; the gap is chunking long text politely and making it barge-in-interruptible
   (the known gap that tool calls aren't interruptible would matter less here since it's
   pure TTS).

7. **Per-tool timeouts / health surfacing.** The SearXNG logs already show engine
   timeouts; surfacing tool health ("web search is slow right now") rather than silent
   degradation would improve trust. Minor, but cheap.

**My overall recommendation:** sequence it as **self-echo refactor (#A1) → reminders
(#1) → briefing (#A2) → Kiwix (#3) → vector-DB background scan (#4) → multi-lingual
(#2)**. The refactor and reminders together unlock the most downstream value; everything
else is comparatively self-contained and can slot in as time allows.
