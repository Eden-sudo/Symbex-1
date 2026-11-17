# damage_probe -- logbook

Every entry: date, question, what was run, numbers, errors, decision. Newest last.

### 2026-11-16: pre-registration
- Question, pairs, the four measures and the rule are in README.md, written before any measurement. Branch
  exp/damage-probe (folder ~/Proyectos/Symbex1_damage, the main folder was busy with merges).
- Ground truth = the tiny_llm runs already trained (runs/ in the main folder, gitignored): no new full training.
- Edén's classic ESP32 checked the same day: no PSRAM (probe firmware: "PSRAM ID read error"), 4 MB flash, 346 KB
  free heap. Not used here; it matters for the on-chip LLM design that this tool should serve.
