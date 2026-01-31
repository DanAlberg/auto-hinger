# Agent Guidelines

## Output hygiene (avoid legacy noise)
- Keep messages and comments present-tense and actionable; avoid legacy/compatibility/deprecation chatter unless it is still required for the user.
- If a feature is removed or replaced, delete or update old outputs so logs don't reference missing code.
- Prefer short, current-state phrasing (example: "Y-band detected." instead of "deprecated"/"legacy" notes).

## Workflow guardrails (for this repo)
- Plan first; no code changes until the user explicitly says "GO".
- Before coding: summarize proposed edits and ask for confirmation.