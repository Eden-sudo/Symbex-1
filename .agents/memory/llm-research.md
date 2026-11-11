# Language models on microcontrollers: research notes

Owners: Obed, Eden. Goal and items: [docs/roadmap.md](../../docs/roadmap.md), "LLM on the chip". Our own probes
(memory bandwidth per tier, 1-bit GEMV kernels) are in PR #2, `benchmarks/edge_llm_probe`.

## Measured by us

microSD on Eden's ESP32-S3 (SDMMC 1-bit, CLK 39 / CMD 38 / D0 40), SanDisk Ultra microSDHC A1 32 GB, raw sector
reads, reported by Eden in PR #12: 4.90 MB/s sequential (128 KB) at 40 MHz, 98 % of the 1-line bus limit; random 4 KB
reads 0.25 MB/s (~17 ms each). With this card and bus, streaming 100 MB of weights per token would take ~20 s per
token sequential, far longer if the reads are random.

## Prior work by others (November 2026), as reported by its authors

- [esp32-ai](https://github.com/slvDev/esp32-ai): 28.9M parameters, 4 bits, ESP32-S3 (8 MB PSRAM, 16 MB flash),
  9.88 tokens/s. ~25M of the parameters are a per-layer embedding table in flash (idea from Gemma 3n); only ~450 bytes
  of it are read per token. TinyStories: short stories, no instructions. MIT, ~4.6k stars, widely covered in the press.
- [p-for-llm](https://github.com/cyfrit/p-for-llm): 180.9M, W1.58A8 (ternary weights, 8-bit activations), mixture of
  experts (29 per layer, top-1) plus per-layer embeddings, ~44 MiB across flash and PSRAM, ESP32-P4, ~9 tokens/s.
  Trained on ~12B tokens on one RTX 5060 Ti. Its author calls the output early and unstable.
- [ESP32s3-LLM-Cluster](https://github.com/Low-Zi-Hong/ESP32s3-LLM-Cluster): Qwen2-0.5B, ternary, 7 ESP32-S3 boards
  over SPI, ~1.3 s per board per token; under-trained, random tokens per its author.
- [esp32-llm](https://github.com/DaveBben/esp32-llm): 260K parameters (llama2.c), 19 tokens/s. Its page does not name
  PSRAM as the bottleneck.
- Raspberry Pi 5 16 GB: Qwen3-30B-A3B at ~8 tokens/s, ~2.7 bits per weight
  ([ByteShape](https://hn.jakechampion.name/reader/https://byteshape.com/blogs/Qwen3-30B-A3B-Instruct-2507)).
- Experts streamed from SSD: llama.cpp discussions
  [#23324](https://github.com/ggml-org/llama.cpp/discussions/23324) and
  [#27149](https://github.com/ggml-org/llama.cpp/discussions/27149); [EdgeMoE](https://arxiv.org/pdf/2308.14352) on a
  Pi 4: loading ~31x slower than compute.
- No measured project streaming LLM weights from microSD on an ESP32, and no fully binary (XNOR) language model on a
  microcontroller, were found.

## Facts checked

- BitNet b1.58: ternary weights, 8-bit activations, trained from scratch, limited non-English support, needs
  bitnet.cpp kernels for speed ([model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)).
- Dynamic Tanh (replaces RMSNorm in Extropic's Z1T): folds into an integer threshold only when the next operation is a
  sign() binarization; not before residuals or attention.
- Extropic Z1T ([article](https://extropic.ai/writing/z1t/)): energy figures are estimates for a probabilistic chip plus
  an FPGA, not silicon measurements; the FPGA takes >95 % of the estimated energy; 0.295 uJ/token without the vocabulary
  projection, ~136.4 uJ/token with it. Not transferable hardware; the fixed small fan-in is an idea worth testing.

## Ceiling to remember

tokens/s <= read bandwidth / bytes of weights read per token. Reduce bytes (fewer bits, fewer weights per token)
before buying faster storage.
