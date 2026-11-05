# other_probes -- single questions

| Script | Question | Result |
|---|---|---|
| `poplut_probe.py` | can the 1-bit weights of the trained mlp_fashion SYMBEX x2 be stored as indices into a small codebook (PopLUT)? | a falsification probe: see `../../mlp/mlp_fashion` (README, PopLUT section) |
| `gpu_bench.py` | every arithmetic (FP32, TF32, FP16, BF16, INT8, 1 bit) on the CUDA GPU, on the same matrix product | printed by the script, a few seconds per size |

```
cd ../../mlp/mlp_fashion && python ../../experiments/other_probes/poplut_probe.py --model runs/symbex_local_s42/model_binarized.pth
python gpu_bench.py --sizes 1024,4096,8192
```
