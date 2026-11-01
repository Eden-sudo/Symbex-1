# SYMBEX-1 documentation

How SYMBEX-1 works, how to use it and what it measured. The [main README](../README.md) has the summary; these pages
go into the details.

| Page | What it explains |
|---|---|
| [How it works](how-it-works.md) | Binary networks, counting bits instead of multiplying, folding everything into integer thresholds, bit-exactness |
| CDT *(coming)* | The compiler: from a PyTorch model to a checked C header, on the PC and on the chip |
| Library *(coming)* | lib_symbex: the layers, the hardware layer of each chip, how to add it to a project |
| Results *(coming)* | Every benchmark in one place, with its charts |

The diagrams are written in [Mermaid](https://mermaid.js.org/) inside the pages (GitHub draws them). Charts live in
[`assets/`](assets/) and are generated from the result files of the benchmarks, so they can always be re-created.
