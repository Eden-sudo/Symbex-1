"""
chip/ -- everything that runs on a real board.

    core.py      boards, the PlatformIO / ESP-IDF project, flashing and the serial port (shared by everything)
    measure.py   tests an exported model on the board: exactness, accuracy, latency per layer, RAM
                 (and the INT8 reference with ESP-NN on the same images)
    kernels.py   kernel benchmark: which arithmetic (XNOR + popcount or AND + count) is faster on THIS chip;
                 the result is saved as a chip profile and the CDT can export with the fastest one
    firmware/    the firmware of each tool, as real .c files
"""
from .core import BOARDS
