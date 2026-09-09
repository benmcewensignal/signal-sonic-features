# signal-sonic-features

Feature conversion worker for Sonic. Re-analyses records to the current analyser version
on its own runner and publishes `out/features-*.jsonl`; the main pipeline imports them
with `python -m sonic.import_features`.

It never writes `sonic.db`. It samples one record per scene-month rather than sweeping in
insertion order, because a sweep silently samples whichever wave was ingested last.

Needs repo secrets `BEATPORT_USERNAME` and `BEATPORT_PASSWORD`.
