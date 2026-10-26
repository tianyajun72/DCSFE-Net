# SpikingScattering Signal Classification



## 1. Overview and current status

The script uses TensorFlow/Keras and Kymatio for multiclass classification of two-channel, one-dimensional signals. It extracts wavelet scattering features separately from each channel, concatenates them, applies a 1D convolution and a dense classifier, and plots training and validation curves.

**The file contains model construction, training, and plotting code, but it is not a complete standalone experiment entry point.** It does not define `Len_signal`, `class_num`, `Train_signal_data`, or `label_data_train`, and contains no data-loading statements. Prepare real data and define these variables before running it; installing dependencies alone does not resolve these omissions.

This README describes the original script above. 

## 2. Suggested environment and dependencies


| Component / pip package | Suggested version | Purpose and usage in the script |
| --- | --- | --- |
| Python | 3.10, 64-bit | Interpreter; use a dedicated Conda environment |
| `tensorflow` | 2.10.1 | Tensor operations, custom layers, and training backend |
| `keras` | 2.10.0 | Model, classifier layers, and callbacks; paired with TensorFlow 2.10 |
| `kymatio` | 0.3.0 | The `kymatio.keras.Scattering1D` wavelet scattering layer |
| `numpy` | 1.23.5 | Arrays, normalization, and plot tick positions |
| `scipy` | 1.10.1 | `scipy.io` is imported; no `.mat` file is loaded in the original script |
| `scikit-learn` | 1.2.2 | `shuffle` and `confusion_matrix` are imported but not called |
| `matplotlib` | 3.7.5 | Confusion-matrix helper and training curves |

`os` is part of Python's standard library, needs no separate installation, and is currently unused. The pip package name for `sklearn` is `scikit-learn`. All third-party packages in the table are required if the original imports are retained, including imports unused later in the script.

Start with CPU checks, then use an NVIDIA GPU if needed. Native Windows CUDA support ends at TensorFlow 2.10; the corresponding legacy GPU configuration uses CUDA 11.2 and cuDNN 8.1. See the [TensorFlow Windows configuration table](https://www.tensorflow.org/install/source_windows#gpu).

Kymatio 0.3.0 officially lists Linux and macOS as supported systems. Run the checks below for the proposed Windows setup; consider WSL2/Linux if platform compatibility issues occur. See the [Kymatio 0.3.0 installation notes](https://pypi.org/project/kymatio/0.3.0/).

## 3. Installation and environment checks

Run these commands in PowerShell with Conda available, or in Anaconda Prompt:

```powershell
conda create -n spiking-scatter python=3.10 -y
conda activate spiking-scatter
python -m pip install --upgrade pip
python -m pip install tensorflow==2.10.1 keras==2.10.0 kymatio==0.3.0 numpy==1.23.5 scipy==1.10.1 scikit-learn==1.2.2 matplotlib==3.7.5
python -m pip check
```

Check imports, then run a scattering-layer forward pass without training data:

```powershell
python -c "import numpy, scipy, sklearn, matplotlib, keras, tensorflow, kymatio; print('Imports OK')"
python -c "import tensorflow as tf; from kymatio.keras import Scattering1D; x=tf.zeros([2, 1000], dtype=tf.float32); s=Scattering1D(J=2, Q=2)(x); print('Scattering shape:', s.shape); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

The second command uses two zero signals of length 1000 solely to check layer construction and computation. Here, 1000 is an example length for this check, not a signal length configured in the original script. The scattering output should be a rank-3 tensor with a batch dimension of 2. An empty GPU list, `[]`, is expected for a CPU setup. Passing these checks does not supply the missing dataset or establish full experimental reproduction.

For optional native Windows GPU use, install a compatible NVIDIA driver and configure CUDA/cuDNN in this environment following the [TensorFlow installation guide](https://www.tensorflow.org/install/pip#windows-native). Skip this step for CPU use:

```powershell
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If TensorFlow reports DLL loading errors on Windows, check the Microsoft Visual C++ runtime requirements in the same installation guide. Neither PyTorch nor MATLAB is required for this script.

## 4. Input requirements and preparation

| Variable | Expected shape / type | Meaning |
| --- | --- | --- |
| `Train_signal_data` | `(N, L, 2)`, preferably real-valued `float32` | N samples, L time points per sample, and 2 channels |
| `label_data_train` | `(N, C)`, preferably `float32` | One-hot labels aligned with the signal samples |
| `Len_signal` | Positive integer L | Signal length, matching the second input dimension |
| `class_num` | Number of classes C | Must match the label width and Softmax output dimension |

The two channels could represent I/Q, but the source does not define their meaning; follow the actual dataset specification. Convert complex signals into two real-valued channels if appropriate, and verify the time and channel axes. The configured `categorical_crossentropy` loss expects categorical targets: do not directly substitute integer labels of shape `(N,)` for one-hot labels. Encode them first, or separately adapt the loss.

The original script does not specify a data format, file path, MATLAB field names, or class names. The following is an **example `.npz` loader to add yourself**. Its path and keys are example conventions and must match a real, prepared dataset:

```python
# Insert after the imports and before the SpikingScattering class definition.
with np.load(r"D:\path\to\your_training_data.npz") as data:
    Train_signal_data = np.asarray(data["Train_signal_data"], dtype=np.float32)
    label_data_train = np.asarray(data["label_data_train"], dtype=np.float32)

assert Train_signal_data.ndim == 3 and Train_signal_data.shape[-1] == 2
assert label_data_train.ndim == 2
assert Train_signal_data.shape[0] == label_data_train.shape[0]
assert np.isfinite(Train_signal_data).all() and np.isfinite(label_data_train).all()
assert np.all((label_data_train == 0) | (label_data_train == 1))
assert np.allclose(label_data_train.sum(axis=1), 1)

Len_signal = int(Train_signal_data.shape[1])
class_num = int(label_data_train.shape[1])
```

You may instead add `sc.loadmat(...)` for a compatible `.mat` file. Inspect its actual field names and dimensions rather than assuming they match this example.

**Definition order matters:** the source declares `def __init__(..., T=Len_signal, ...)`. Python evaluates default arguments when defining the function, so `Len_signal` must exist before the `SpikingScattering` class is defined. Otherwise, a `NameError` occurs even before model construction. Changing the default to `T=None` is an optional refactoring choice; this documentation task does not modify the source.

After preparing data and defining the missing variables, run from PowerShell:

```powershell
conda activate spiking-scatter
Set-Location -LiteralPath 'D:\RIS数据集训练\05B 稿件返修\（TCCN）'
python '.\Main_SpikingScatter_ESR - 副本.py'
```

Keep the quotes because the filename contains spaces. The original script has no `--data` argument and no `if __name__ == '__main__'` guard; importing the file also executes its top-level model construction and training statements.

## 5. Main code workflow

### 5.1 Helper functions

- `F_normalize(dt)` computes the mean and standard deviation along `axis=0` and returns `(dt - mu) / sigma`. It is not called in the current training path. For `(N, L, 2)` input, this operates across samples. Handle zero standard deviations before using it, and estimate normalization statistics using the training portion only.
- `plot_confusion_matrix(cm, labels=...)` plots an existing confusion matrix with a color bar, class ticks, and cell counts. Rows represent true classes and columns predicted classes. The helper does not perform prediction or calculate the matrix, and is not called in the current script.

### 5.2 Scattering layer and classifier

```text
Input (batch, L, 2)
  ├─ Channel 0 → Scattering1D(J=2, Q=2) → (batch, K, Ls)
  └─ Channel 1 → Scattering1D(J=2, Q=2) → (batch, K, Ls)
           ↓ Transpose each output to (batch, Ls, K)
  Concatenate along the last axis → (batch, Ls, 2K)
           ↓ Conv1D(filters=16, kernel_size=3, strides=2)
  Flatten
           ↓ Dense(128, ReLU) → Dropout(0.2)
           ↓ Dense(64, ReLU)  → Dropout(0.2)
           ↓ Dense(C, Softmax)
  C class probabilities per sample
```

`K` is the number of scattering coefficients per channel, and `Ls` is the scattering output's temporal length. Their exact values depend on input length and scattering parameters.

`SpikingScattering.build()` reads the input length and creates the scattering and convolution layers. Its `call()` method processes each channel, transposes the outputs, concatenates them, and applies the convolution. `Conv1D` does not specify an activation, so it uses a linear activation, with default `valid` padding. `model.summary()` prints the assembled model.

`J=2` controls the maximum scattering scale (`2**J`), and `Q=2` specifies the first-order wavelets per octave. The script does not set `max_order`; Kymatio 0.3.0 defaults to second-order scattering. Its Keras frontend infers the input length during building, so the current `Scattering1D(J=..., Q=...)` call requires no additional `shape` argument. See the [Kymatio Keras API](https://www.kymat.io/codereference.html#kymatio.keras.Scattering1D).

The custom layer's `self.T` records input length but is never passed to Kymatio. It has a different meaning from Kymatio's own `T` parameter, which controls the temporal support of its low-pass filter.

Although the class name contains “Spiking”, the implemented operations are wavelet scattering and an ordinary 1D convolution. The file contains no explicit threshold firing, LIF neuron, or binary spike-encoding operator. Describe the implementation according to these actual operations.

### 5.3 Training configuration

| Setting | Current value |
| --- | --- |
| Optimizer | Adam; no explicit learning rate |
| Loss | `categorical_crossentropy` |
| Metric | `accuracy` |
| Batch size | `BATCH_SIZE = 128` |
| Maximum epochs | `EPOCH = 50` |
| Validation fraction | `validation_split = 0.3` |
| Early stopping | Monitor `val_loss`, with `patience = 5` |

`model.fit()` returns `hist`, holding per-epoch training and validation loss and accuracy. Early stopping can end training before 50 epochs. Since `restore_best_weights=True` is not set, training retains the weights from the stopping point. See the [Keras EarlyStopping documentation](https://keras.io/2/api/callbacks/early_stopping/).

For array inputs, Keras reserves roughly the last 30% of supplied samples for validation before shuffling the training portion. This is not a stratified random split. If samples are ordered by class or acquisition sequence, design an appropriate split first. See the [Keras training API](https://keras.io/2/api/models/model_training_apis/).

## 6. Outputs and reproduction notes

- Successful training prints the model summary and training metrics; `model` and `hist` remain in memory.
- The script creates two figures: training/validation loss and training/validation accuracy. Its `plt.savefig(...)` and `plt.show()` calls are commented out. A regular terminal run therefore saves no plots automatically and may display no figure window; uncomment or add the required calls if needed.
- There is no `model.save()` call, and the imported `ModelCheckpoint` is unused. The original script does not automatically export models, weights, or training history files.
- There is no separate test-set evaluation, prediction, or confusion-matrix calculation. The confusion-matrix plotting function is an available helper only.
- Imported `AveragePooling1D` and `AveragePooling2D` are not part of the model; do not include pooling layers in an architecture description based on imports alone.
- No random seed is set, so repeated training results may differ. Full reproduction also requires documented data splits, preprocessing, class mappings, environment versions, and random seeds.
- The confusion-matrix helper uses Times New Roman. Matplotlib may substitute a font or issue a missing-font message if it is unavailable.


