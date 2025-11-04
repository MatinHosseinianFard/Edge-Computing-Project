# Edge Computing Project – Human Activity Recognition on Tiny Devices

## Overview
This project implements a multi-resolution fusion approach for Human Activity Recognition (HAR) from video data on resource-constrained edge devices. The method is inspired by a two-stream architecture that processes video frames at different resolutions to reduce computational cost while maintaining recognition accuracy.

The system uses:
- **Context stream**: Processes low-resolution full-frame video for general motion understanding.
- **Fovea stream**: Processes high-resolution cropped regions around areas of interest to capture fine details.

To deploy the model efficiently on microcontrollers, post-training quantization (PTQ) and quantization-aware training (QAT) techniques are applied to compress the model while preserving accuracy.

---

## Technologies Used
- **Deep Learning Frameworks**: TensorFlow, TensorFlow Lite, TensorFlow Lite Micro  
- **Model Optimization Techniques**:  
  - Post-Training Quantization (PTQ)  
  - Quantization-Aware Training (QAT)  
- **Edge Hardware**: ESP32-S3 microcontroller  
- **Datasets**:  
  - KTH Action Dataset  
  - UCF11 (partially evaluated)  
- **Programming Languages**: Python, C/C++ (for microcontroller deployment)

---

## Applications
- Real-time activity recognition on wearable or embedded devices  
- Surveillance and smart home automation  
- Health monitoring and elderly fall detection  
- Edge AI systems with low latency and minimal energy consumption

---

## Results

### Performance on KTH Dataset (Full Precision Model)
| Model            | Precision | Recall | F1 Score |
|------------------|-----------|--------|----------|
| KTH (32×16)      | ~99%      | ~99%   | ~99%     |
| KTH (64×32)      | ~99%      | ~99%   | ~99%     |

### After Quantization (PTQ and QAT)
| Method | Accuracy | Model Size Reduction |
|--------|----------|------------------------|
| PTQ    | ~98–99%  | Up to 70–80% smaller   |
| QAT    | ~98–99%  | Similar size, higher stability |

### Inference on ESP32-S3
| Model            | Inference Time (ms) |
|------------------|----------------------|
| KTH (32×16, QAT) | ~16 ms               |
| KTH (64×32, QAT) | ~64 ms               |

The quantized models achieved high accuracy while being lightweight enough to run efficiently on microcontrollers with limited memory and processing power.

---

## Key Contributions
- Implemented a dual-resolution architecture optimized for video-based HAR on edge devices.
- Achieved high accuracy while significantly reducing model size using PTQ and QAT.
- Deployed the model on an ESP32-S3 microcontroller for real-time inference.
- Demonstrated that activity recognition from video is feasible on tiny edge devices without cloud dependence.

---

## Limitations and Future Work
- The UCF11 dataset was only partially evaluated due to dataset complexity.
- Energy consumption measurements were not fully conducted due to hardware limitations.
- Future improvements may include:
  - Integration of recurrent layers (LSTM/RNN) for temporal modeling  
  - Model pruning and knowledge distillation  
  - Support for more datasets and real-world scenarios  
  - Generalized models for multiple activity recognition tasks

---

## Reference
Based on the research paper:  
*“A multi-resolution fusion approach for human activity recognition from video data in tiny edge devices,” Information Fusion, 2023.*

