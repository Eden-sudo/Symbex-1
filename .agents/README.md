# Shared project memory for agents

What every agent working on SYMBEX-1 (Eden's, Obed's, Abrahan's and the reviewer) should know and is not obvious from
the code. Start at [AGENTS.md](../AGENTS.md).

| File | Contents |
|---|---|
| [memory/team.md](memory/team.md) | who owns which goal and what each person usually does |
| [memory/conventions.md](memory/conventions.md) | branches, commits, PRs, `UPDATES.md`, measurement rules, what is not published |
| [memory/fomo.md](memory/fomo.md) | vision: what exists and the gaps for more classes and a camera |
| [memory/llm-research.md](memory/llm-research.md) | language models: our measurements, prior work by others, facts checked |

Rules for this folder:

- Write facts with their evidence (file, PR number, JSON, link) and the date when they can change. Mark what is
  reported by others or not measured by us.
- Update a file in the same pull request as the work that changes it, like `docs/roadmap.md`.
- No secrets, tokens, personal paths or private data; this folder is public.
- The roadmap is `docs/roadmap.md` and the per-PR history is `UPDATES.md`; do not copy them here.
