// Generado por compiler_core/export/plain.py:generar_wiring_cpp() -- no editar a mano.
// main.cpp debe #include este archivo y usar bloques[] / n_bloques / nombres_bloques[]
// en vez de declarar los suyos.

#define RES_STEM 32

struct SymbexResidualBlock bloque_layer1_0 = { HACER_CONV(LAYER1_0_CONV1, 32, 32), HACER_CONV(LAYER1_0_CONV2, 32, 32), NULL, NULL, false };
struct SymbexResidualBlock bloque_layer1_1 = { HACER_CONV(LAYER1_1_CONV1, 32, 32), HACER_CONV(LAYER1_1_CONV2, 32, 32), NULL, NULL, false };
struct SymbexResidualBlock bloque_layer2_0 = { HACER_CONV(LAYER2_0_CONV1, 32, 32), HACER_CONV(LAYER2_0_CONV2, 16, 16), NULL, NULL, true };
struct SymbexResidualBlock bloque_layer2_1 = { HACER_CONV(LAYER2_1_CONV1, 16, 16), HACER_CONV(LAYER2_1_CONV2, 16, 16), NULL, NULL, false };
struct SymbexResidualBlock bloque_layer3_0 = { HACER_CONV(LAYER3_0_CONV1, 16, 16), HACER_CONV(LAYER3_0_CONV2, 8, 8), NULL, NULL, true };
struct SymbexResidualBlock bloque_layer3_1 = { HACER_CONV(LAYER3_1_CONV1, 8, 8), HACER_CONV(LAYER3_1_CONV2, 8, 8), NULL, NULL, false };
struct SymbexResidualBlock bloque_layer4_0 = { HACER_CONV(LAYER4_0_CONV1, 8, 8), HACER_CONV(LAYER4_0_CONV2, 4, 4), NULL, NULL, true };
struct SymbexResidualBlock bloque_layer4_1 = { HACER_CONV(LAYER4_1_CONV1, 4, 4), HACER_CONV(LAYER4_1_CONV2, 4, 4), NULL, NULL, false };

void symbex_preparar_bloques() {
    bloque_layer1_0.beta_rsign1 = LAYER1_0_CONV1_beta_rsign_entrada;
    bloque_layer1_0.beta_rsign2 = LAYER1_0_CONV2_beta_rsign_entrada;

    bloque_layer1_1.beta_rsign1 = LAYER1_1_CONV1_beta_rsign_entrada;
    bloque_layer1_1.beta_rsign2 = LAYER1_1_CONV2_beta_rsign_entrada;

    bloque_layer2_0.beta_rsign1 = LAYER2_0_CONV1_beta_rsign_entrada;
    bloque_layer2_0.beta_rsign2 = LAYER2_0_CONV2_beta_rsign_entrada;
    static const SymbexDspParamsSimple ds_layer2_0 = { LAYER2_0_DOWNSAMPLE_escala, LAYER2_0_DOWNSAMPLE_shift_bn, SYMBEX_Q_ACT, SYMBEX_Q_MULT };
    bloque_layer2_0.downsample_pesos = LAYER2_0_DOWNSAMPLE_pesos;
    bloque_layer2_0.beta_rsign_downsample = LAYER2_0_DOWNSAMPLE_beta_rsign_entrada;
    bloque_layer2_0.downsample_dsp = ds_layer2_0;
    bloque_layer2_0.downsample_kernel = LAYER2_0_DOWNSAMPLE_KERNEL;
    bloque_layer2_0.downsample_padding = LAYER2_0_DOWNSAMPLE_PADDING;

    bloque_layer2_1.beta_rsign1 = LAYER2_1_CONV1_beta_rsign_entrada;
    bloque_layer2_1.beta_rsign2 = LAYER2_1_CONV2_beta_rsign_entrada;

    bloque_layer3_0.beta_rsign1 = LAYER3_0_CONV1_beta_rsign_entrada;
    bloque_layer3_0.beta_rsign2 = LAYER3_0_CONV2_beta_rsign_entrada;
    static const SymbexDspParamsSimple ds_layer3_0 = { LAYER3_0_DOWNSAMPLE_escala, LAYER3_0_DOWNSAMPLE_shift_bn, SYMBEX_Q_ACT, SYMBEX_Q_MULT };
    bloque_layer3_0.downsample_pesos = LAYER3_0_DOWNSAMPLE_pesos;
    bloque_layer3_0.beta_rsign_downsample = LAYER3_0_DOWNSAMPLE_beta_rsign_entrada;
    bloque_layer3_0.downsample_dsp = ds_layer3_0;
    bloque_layer3_0.downsample_kernel = LAYER3_0_DOWNSAMPLE_KERNEL;
    bloque_layer3_0.downsample_padding = LAYER3_0_DOWNSAMPLE_PADDING;

    bloque_layer3_1.beta_rsign1 = LAYER3_1_CONV1_beta_rsign_entrada;
    bloque_layer3_1.beta_rsign2 = LAYER3_1_CONV2_beta_rsign_entrada;

    bloque_layer4_0.beta_rsign1 = LAYER4_0_CONV1_beta_rsign_entrada;
    bloque_layer4_0.beta_rsign2 = LAYER4_0_CONV2_beta_rsign_entrada;
    static const SymbexDspParamsSimple ds_layer4_0 = { LAYER4_0_DOWNSAMPLE_escala, LAYER4_0_DOWNSAMPLE_shift_bn, SYMBEX_Q_ACT, SYMBEX_Q_MULT };
    bloque_layer4_0.downsample_pesos = LAYER4_0_DOWNSAMPLE_pesos;
    bloque_layer4_0.beta_rsign_downsample = LAYER4_0_DOWNSAMPLE_beta_rsign_entrada;
    bloque_layer4_0.downsample_dsp = ds_layer4_0;
    bloque_layer4_0.downsample_kernel = LAYER4_0_DOWNSAMPLE_KERNEL;
    bloque_layer4_0.downsample_padding = LAYER4_0_DOWNSAMPLE_PADDING;

    bloque_layer4_1.beta_rsign1 = LAYER4_1_CONV1_beta_rsign_entrada;
    bloque_layer4_1.beta_rsign2 = LAYER4_1_CONV2_beta_rsign_entrada;

}

struct SymbexBloqueEntry { struct SymbexResidualBlock* bloque; uint32_t n_salida; };
struct SymbexBloqueEntry bloques[] = {
    {&bloque_layer1_0, (uint32_t)LAYER1_0_CONV2_OUT_CH * 32 * 32},
    {&bloque_layer1_1, (uint32_t)LAYER1_1_CONV2_OUT_CH * 32 * 32},
    {&bloque_layer2_0, (uint32_t)LAYER2_0_CONV2_OUT_CH * 16 * 16},
    {&bloque_layer2_1, (uint32_t)LAYER2_1_CONV2_OUT_CH * 16 * 16},
    {&bloque_layer3_0, (uint32_t)LAYER3_0_CONV2_OUT_CH * 8 * 8},
    {&bloque_layer3_1, (uint32_t)LAYER3_1_CONV2_OUT_CH * 8 * 8},
    {&bloque_layer4_0, (uint32_t)LAYER4_0_CONV2_OUT_CH * 4 * 4},
    {&bloque_layer4_1, (uint32_t)LAYER4_1_CONV2_OUT_CH * 4 * 4},
};
const int n_bloques = 8;
const char* nombres_bloques[] = {"layer1_0", "layer1_1", "layer2_0", "layer2_1", "layer3_0", "layer3_1", "layer4_0", "layer4_1"};
