# Host MLP example

This host check compares the demo images against their reference results; it is
not an accuracy measurement for a trained model.

From the repository root, build with:

```sh
gcc -std=c99 -Wall -Wextra -Werror -I Symbex1_PIO/lib/lib_symbex/include -I Symbex1_PIO/src Symbex1_PIO/lib/lib_symbex/examples/host_mlp/main.c Symbex1_PIO/lib/lib_symbex/src/layers/conv/symbex_bool.c Symbex1_PIO/lib/lib_symbex/src/hal/generic/symbex_hal_impl.c -o /tmp/symbex-host-mlp
```

Run `/tmp/symbex-host-mlp`. Currently it prints `OK 8/8` and exits 0 on
success; the first mismatch is printed to stderr and exits 1.
