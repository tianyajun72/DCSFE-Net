
# DCSFE-Net: Physics-Aware Electromagnetic Signal Recognition for RIS-Aided Industrial IoT

<p align="center">
  <strong>Official Implementation and Experimental Results</strong>
</p>

<p align="center">
  A Dual-Branch Spiking-Scattering Fusion Framework for 6G RIS-Assisted Industrial Internet of Things
</p>

---

## Overview

Electromagnetic Signal Recognition (ESR) is an important enabling technology for spectrum management, interference identification, and reliable wireless communications in next-generation industrial Internet of Things (IIoT) networks. In 6G Reconfigurable Intelligent Surface (RIS)-assisted environments, electromagnetic signals may experience severe multipath propagation, non-line-of-sight (NLOS) blockage, reflection-induced waveform distortions, and noise interference. These practical impairments make reliable signal recognition challenging for conventional data-driven neural networks.

To address these challenges, this project presents **DCSFE-Net (Dual-Branch Spiking-Scattering Fusion Network)**, a physics-aware electromagnetic signal recognition framework designed for RIS-assisted industrial wireless environments. The proposed architecture integrates wavelet scattering transforms with spiking neural networks (SNNs), combining deformation-stable signal representations with temporal spike-based feature encoding.

By processing the in-phase (I) and quadrature (Q) components through complementary branches and adaptively fusing their learned representations, DCSFE-Net aims to improve signal discriminability, noise robustness, and inference efficiency.

Unlike approaches evaluated exclusively in idealized simulated channels, this study incorporates a physical RIS measurement platform to investigate recognition performance under realistic electromagnetic propagation conditions.

## Key Contributions

### 1. Physics-Aware Electromagnetic Signal Recognition

The proposed framework incorporates signal-processing principles associated with electromagnetic propagation into the feature extraction architecture. Wavelet scattering transforms provide multiscale representations with deformation stability, helping preserve discriminative signal characteristics under propagation-induced distortions.

### 2. Spiking-Scattering Feature Learning

DCSFE-Net combines two complementary mechanisms:

- **Wavelet Scattering Transform:** Extracts multiscale spectral-temporal features using cascaded wavelet convolutions, nonlinear modulus operations, and low-pass filtering.
- **Spiking Neural Network:** Encodes temporal information through membrane-potential dynamics, threshold-triggered spike generation, and subtractive soft-reset operations.

This combination is designed to retain useful signal structures while reducing the influence of weak or noise-corrupted temporal components.

### 3. Dual-Branch I/Q Feature Fusion

The network processes the in-phase and quadrature components through two complementary feature extraction branches. Their learned representations are subsequently combined through an adaptive fusion mechanism, allowing the classifier to exploit complementary information contained in the complex-valued received signal.

### 4. Physical RIS Testbed Validation

The experimental framework integrates a programmable RIS metasurface with a professional RF signal generation and measurement chain. The physical platform enables the collection and evaluation of electromagnetic signals under RIS-assisted propagation, including NLOS configurations and reflection-induced channel impairments.

### 5. Recognition Performance and Computational Efficiency

The study evaluates DCSFE-Net against representative deep-learning-based ESR methods, including LSTM, TCNet, ComplexNet, and other signal recognition baselines.

Recognition accuracy, computational complexity, inference latency, and the contributions of individual architectural components are investigated through comparative and ablation experiments.

## Network Architecture

The proposed DCSFE-Net consists of four main functional components:

1. **Wavelet Scattering Transform:** Extracts deformation-stable, multiscale signal features.
2. **Spiking Neural Network:** Encodes temporal dynamics into spike-based representations.
3. **Dual-Branch Fusion Module:** Integrates complementary I/Q branch features for discriminative representation learning.
4. **Decision-Making Module:** Uses a multilayer perceptron (MLP) classifier to predict the electromagnetic signal class.

The overall processing pipeline can be summarized as follows:

```text
Received RF Signal
        |
        v
   I/Q Decomposition
        |
        v
 Dual-Branch Feature Extraction
        |
        +-------------------------+
        |                         |
        v                         v
 Wavelet Scattering       Spiking Temporal
 Feature Learning         Feature Encoding
        |                         |
        +------------+------------+
                     |
                     v
             Feature Fusion
                     |
                     v
              MLP Classifier
                     |
                     v
          Predicted Signal Class
```

The central idea is to combine the robustness of physics-inspired signal representations with the temporal modeling capability of spiking neural networks, rather than relying solely on conventional convolutional or recurrent feature learning.

## Experimental Setup

### Signal Classes

The experimental dataset covers seven electromagnetic signal formats:

- BPSK
- QPSK
- 8PSK
- 2FSK
- 16APSK
- 16QAM
- 64QAM

### Dataset Configuration

| Parameter | Configuration |
|---|---|
| Number of signal classes | 7 |
| SNR range | -10 dB to 20 dB |
| SNR interval | 2 dB |
| Number of SNR levels | 16 |
| Samples per class per SNR level | 2,250 |
| Dataset partition | 60% training / 20% validation / 20% testing |
| Signal representation | In-phase (I) and quadrature (Q) components |

The dataset configuration corresponds to 252,000 signal instances across the seven classes and sixteen SNR levels, based on the sample counts reported in the paper.

### Physical Measurement Platform

The real-world measurement setup includes:

- A programmable RIS metasurface with a diagonal dimension of approximately 0.6 m.
- A 16 × 16 reflective-element array comprising 256 elements.
- A Rohde & Schwarz SMBV100B vector signal generator.
- A Rohde & Schwarz FSW spectrum analyzer.
- A transmission configuration centered at 4.8 GHz with a signal bandwidth of 100 MHz.

This platform supports experimental investigation of RIS-assisted signal propagation and provides a hardware-based foundation for evaluating electromagnetic signal recognition.

## Experimental Results

The paper reports the following representative results:

- **Recognition improvement at low SNR:** DCSFE-Net achieves a reported 6.7% recognition-accuracy improvement over conventional baseline methods at an SNR of 4 dB.
- **Comparison with sequential and convolutional baselines:** At 10 dB SNR, the reported accuracy gains over LSTM and TCNet are 11.28% and 16.35%, respectively.
- **High-SNR classification:** The confusion matrix reports 100% classification accuracy at 18 dB SNR for the evaluated test configuration.
- **Computational complexity:** The model contains 263,383 trainable parameters, with a reported training time of approximately 7 seconds per epoch.
- **Inference efficiency:** The reported testing time is 0.03348 ms under the paper's experimental evaluation setup.

Additional experiments investigate the effects of spiking encoding, wavelet scattering, scattering coefficients, SNN convolution stride, RIS orientation, signal observation length, training-data ratio, and dual-branch feature fusion.

Together, these results demonstrate the potential of combining physics-aware scattering representations and spiking temporal encoding for efficient electromagnetic signal recognition in RIS-assisted wireless environments.

## Implementation Environment

The reported implementation environment includes:

| Component | Version or Specification |
|---|---|
| Deep learning framework | TensorFlow 2.10.0 |
| CUDA | 11.8 |
| cuDNN | 8.4.1 |
| Signal preprocessing | MATLAB R2024a |
| GPU | NVIDIA GeForce RTX 4060, 8 GB memory |
| Operating system | Windows 11 Professional |

Actual reproduction may require adapting software dependencies, dataset paths, and hardware configurations to the available environment.

## Experimental Scope and Reproducibility

The low-SNR experiments use controlled additive white Gaussian noise (AWGN) introduced into signals collected using the physical RIS measurement platform.

Accordingly, these results characterize performance under controlled Gaussian-noise conditions combined with real-world signal acquisition. They should not be interpreted as a comprehensive evaluation under all forms of measured industrial interference, such as site-specific colored or impulsive noise.

For reproducibility, the exact training procedure, preprocessing configuration, model checkpoints, and experimental data should be used where available in this repository. Results may vary with the software environment, hardware platform, random initialization, and evaluation protocol.

## Acknowledgment

This work investigates the integration of physics-aware signal processing, spiking neural networks, and programmable electromagnetic environments for robust signal recognition in emerging 6G industrial IoT systems.
