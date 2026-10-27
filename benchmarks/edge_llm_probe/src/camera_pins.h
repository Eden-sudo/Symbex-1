/*
 * camera_pins.h -- OV2640/OV5640 pin maps of the two supported boards. Copied from the CameraWebServer example of the
 * Arduino core (libraries/ESP32/examples/Camera/CameraWebServer/camera_pins.h, arduino-esp32 2.0.17, models
 * CAMERA_MODEL_AI_THINKER and CAMERA_MODEL_ESP32S3_EYE). For another board, add a block with its pins.
 *
 * The Freenove ESP32-S3-WROOM CAM is documented as using the same GPIOs as the ESP32-S3-EYE map; that is NOT checked
 * here: compare with your board's schematic before trusting a "camera: fail" on it.
 */
#ifndef PROBE_CAMERA_PINS_H
#define PROBE_CAMERA_PINS_H

#if defined(PROBE_CAMERA_AI_THINKER)

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#elif defined(PROBE_CAMERA_ESP32S3_EYE)

#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM      4
#define SIOC_GPIO_NUM      5
#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM        8
#define Y3_GPIO_NUM        9
#define Y2_GPIO_NUM       11
#define VSYNC_GPIO_NUM     6
#define HREF_GPIO_NUM      7
#define PCLK_GPIO_NUM     13

#endif

#endif
