#include <stdio.h>
#include "model_symbex.h"
#include "demo_input.h"

int main(void)
{
    (void)symbex_mlp_steps;
    for (int i = 0; i < DEMO_N_IMAGES; ++i) {
        int got = symbex_mlp_infer(demo_images[i]);
        if (got != demo_expected[i]) {
            fprintf(stderr, "Mismatch image %d: expected %d, got %d\n",
                    i, demo_expected[i], got);
            return 1;
        }
    }
    printf("OK %d/%d\n", DEMO_N_IMAGES, DEMO_N_IMAGES);
    return 0;
}
