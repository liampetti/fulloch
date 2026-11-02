# Deep Think replay fixtures

Keep at most five `*.evidence.json` files here. Use temporary output paths for
replay results; replace an existing fixture deliberately when better evidence is
available, rather than keeping every run. Update fixture-dependent tests when a
fixture is replaced.

| Fixture | Coverage | Origin |
| --- | --- | --- |
| `synthetic-finance.evidence.json` | Quote, currency and quote time | Fabricated provider-shaped record; not market data. |
| `synthetic-travel.evidence.json` | Flight route, dates, fare and itinerary | Fabricated provider-shaped record; not a real offer. |
| `synthetic-ha-history.evidence.json` | Home Assistant temperature history and units | Fabricated history; no household data. |

The legacy saved research fixtures in this directory are not pytest inputs.
Do not add tests that read personal research, conversation history or live
provider responses; create artificial examples instead.

The synthetic fixtures check offline record handling, not whether tools can
retrieve real data or whether the report is useful. Validate real provider and
Home Assistant behavior separately with the scenarios in `dev/test-scenarios.md`.
Do not add real private household or conversation history to the repository.
