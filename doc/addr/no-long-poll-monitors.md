# Decision: no long-poll monitors

## Status
Accepted.

## Context
In a `/loop`-driven workflow, there is a temptation to monitor a long-running
external state (CI, PR review, a remote queue) by spinning up a `while true`
shell loop that polls a status endpoint on a fixed interval. This pattern
appears in `Monitor` invocations like:

```sh
prev=...; while true; do cur=$(...); if [ "$cur" != "$prev" ]; then ...; fi; sleep 60; done
```

## Decision
**Do not use long-poll monitors** in this repo. Use event-driven waiters
instead — fall back to a single `gh pr view` / `gh run watch` / `gh api .../events`
query that exits on the first signal, or to a `ScheduleWakeup` heartbeat when
no event source exists.

## Rationale
1. **API hygiene.** Every poll counts against the rate limit. GitHub's REST
   limits are 5000/h authenticated; a 60 s poll burns 60 calls/h per workflow
   for nothing when no state change has happened.
2. **Wasteful even when cheap.** A long-poll that fires only when something
   changed is identical in user experience to a one-shot poll that exits on
   the first signal. The `while true; do ... sleep N` pattern adds nothing.
3. **Noisy under fleet load.** When multiple `/loop` instances run in
   parallel (e.g. babysitting several PRs), each one independently polls the
   same endpoints. This is unnecessary background noise on the wire.
4. **Already standardized.** The available primitives are:
   - `gh pr view <n>` for one-shot PR state fetch.
   - `gh run watch <run-id>` for CI completion (waits, then exits).
   - `gh api repos/<owner>/<repo>/issues/<n>/events?since=<ts>` for incremental
     comment/review deltas (one-shot query, not a loop).
   - `ScheduleWakeup` for time-driven heartbeats when no event source exists.
   These cover the same surface without polling.
5. **Stale-signal risk.** A long-poll that polls `prev_state` vs `cur_state`
   can miss transitions if the API ever returns a transient value, or can
   double-fire on a non-change that happens to differ in representation. A
   one-shot query on a known transition is unambiguous.

## What to use instead

| Scenario | Use |
|---|---|
| Wait for PR to merge / close | `gh pr view <n> --jq .state` once, or `gh pr checks <n>` to wait on CI |
| Wait for CI to complete | `gh run watch <run-id>` (exits on completion) |
| Wait for a new review comment | One-shot `gh api .../events?since=<last_seen_ts>` |
| Wait for any state change (no event source) | `ScheduleWakeup` with a long fallback (1200–1800 s) |

## When this rule applies
Any `Monitor` invocation whose command body is `while true; do ... sleep N` or
equivalent polling loops is forbidden. A `Monitor` is allowed only when its
command exits on the first event of interest (log tail, websocket, `gh run
watch`, etc.).

## History
- 2026-06-28 — captured after a `/loop` session used a long-poll monitor for
  PR #57 review status. Replaced with one-shot queries + `ScheduleWakeup`
  heartbeat.