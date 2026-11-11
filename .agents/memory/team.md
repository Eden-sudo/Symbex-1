# Team and ownership

| Person | GitHub | Owns | Usual work (from their pull requests) |
|---|---|---|---|
| Carlos Duarte (Eden) | [Eden-sudo](https://github.com/Eden-sudo) | framework, language models | maintainer and merges; C kernels and HALs, CDT and export, re-measuring benchmarks on the boards against TFLite INT8, bilingual docs and `UPDATES.md`; hosts the shared ESP32-S3 |
| Obed | [Obed0101](https://github.com/Obed0101) | language models, framework | test suite, low-bit baselines (INT4 / INT2), validation split, `benchmarks/edge_llm_probe` (PR #2), team coordination (PR #12), docs |
| Abrahan | [abrahandejesusaparicio-code](https://github.com/abrahandejesusaparicio-code) | vision (FOMO) | `benchmarks/conv2d/fomo_voc_person`: sizing by validation F1, training, chip measurement on his classic ESP32, results JSON and README tables |

Goals and owners: [docs/roadmap.md](../../docs/roadmap.md), "The three main goals". Anyone can contribute to any
goal; the owner reviews changes in their area. Reviews are posted by MendCode (`.mendcode/review.yaml`); the author
asks for a re-review after each push.
