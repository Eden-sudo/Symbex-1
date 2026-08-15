# SYMBEX-1 (Symbolic Bit Expansion 1-bit Engine)

**SYMBEX-1** es una arquitectura de Edge-AI e inferencia neuronal ultraligera diseñada para microcontroladores de 8-bits con recursos estáticos restringidos (como SRAM limitada y sin Unidad de Punto Flotante).

Desarrollado como un proyecto de investigación, este motor permite la ejecución de redes neuronales complejas (como generación de trayectorias y visión artificial básica) utilizando exclusivamente operadores lógicos a nivel de bits.

## 🧠 Características Principales

*   **Bit-Slicing (K=3):** Cuantización residual que divide los pesos en tres máscaras lógicas (MSB, MID, LSB), ejecutando la inferencia mediante compuertas `XNOR` y rutinas de `POPCOUNT` para evitar operaciones de punto flotante.
*   **Outlier-Aware Abstraction Layer:** Una capa lógica dispersa y superpuesta que aísla valores de activación masivos, permitiendo comprimir la red hasta en un 81% sin sufrir caídas en la precisión matemática.
*   **Compilador CDT (Compilador Dinámico de Topología):** Middleware en Python (`symbex_compiler.py`) que perfila redes entrenadas en PyTorch, calcula la varianza topológica y ensancha automáticamente la red binarizada compensando la pérdida de precisión sin requerir memoria dinámica.
*   **Bucle Autoregresivo Estático:** Sistema de ventana deslizante (*Sliding Window*) libre de fragmentación de memoria (Zero `malloc`/`new`), diseñado para generar trayectorias continuas en hardware embebido.

## ⚙️ Estructura del Proyecto

*   `/lib_symbex/`: Librería principal en C++ lista para ser importada en Arduino IDE o simulada en GCC.
*   `/symbex_training/`: Entorno en Python (PyTorch) que contiene el Compilador CDT para la destilación del conocimiento y generación del archivo de cabecera de pesos.

## 👨‍💻 Autores
Investigación y desarrollo de arquitectura a cargo de:
*   **Carlos Eden Duarte**
*   **Bryan Mendez**

## 📄 Licencia
Este proyecto está bajo la Licencia **GNU Affero General Public License v3.0 (AGPLv3)**. Consulta el archivo `LICENSE` para más detalles.
