# Malayalam-Regional-Dialect-Recognition
This project aims to enhance Automated Speech Recognition (ASR) for the Malayalam language by focusing on its regional dialects. Utilizing advanced machine learning techniques and a specially curated dataset, this model seeks to significantly improve dialect recognition accuracy, making voice-driven applications more accessible to speakers from different regions of Kerala.

**Note** -  The dataset is not provided in this repository because it is currently being used for an ongoing journal paper. Once the paper is published and the dataset is cleared for public release, it will be updated in this repository with the dataset.

## Authors 
1. Thanushri Madhuraj
2. Aadya Goel
3. Perarapu Lakshmi Sri
4. M R Madhumita

## Dataset 
The primary objective of creating this corpus is to facilitate research into the recognition and analysis of regional dialects in Malayalam speech, focusing on four major dialects corresponding to the districts of Kozhikode, Trissur, Trivandrum, and Kottayam. This initiative addresses the gap in available resources for studying dialected speech in the Malayalam language, which is critical for developing more inclusive and effective speech recognition systems. 


### Hours of Data per Dialect
Below is a table detailing the hours of data collected for each dialect:

| Dialect    | Hours of Data |
|------------|---------------|
| Trissur    | 5.0           |
| Kozhikode  | 5.0           |
| Kottayam   | 5.0           |
| Trivandrum | 3.2           |

*Table 1: Hours of Data per dialect*

### Data Preprocessing
The dataset, collected from diverse sources such as YouTube and cinematic productions, often contains extraneous auditory elements like background music and ambient noise. This section outlines the sophisticated preprocessing techniques employed to prepare the audio data for machine learning.

### Data Collection
The initial phase of data acquisition involves identifying authentic regional content on YouTube, which includes specific channels and videos featuring native speakers from designated dialect regions. URLs from these videos are extracted and processed through YTMP3, an online tool that converts YouTube videos to MP3 format. Subsequently, the MP3 files are converted to WAV format using CloudConvert to facilitate further audio processing.

## Methodology 

<p align="center">
  <img src="https://github.com/Lakshmisri01/Malayalam-Regional-Dialect-Recognition/assets/114591852/627e6b30-4fe4-4de7-8087-adf549fd3a12" alt="Proposed Model for Dialect Classification">
  <br>
  <strong>The Proposed Model for Dialect Classification</strong>
</p>

The methodology employed in this study revolves around leveraging YAMNet, a pretrained deep learning model developed by Google, for the purpose of audio classification. YAMNet is specifically designed to identify a diverse array of sound events within audio recordings, built upon the TensorFlow framework and incorporating the MobileNetV1 architecture to ensure efficiency and scalability in audio classification tasks. One of its key functionalities lies in extracting high-level features, particularly the 1,024-dimensional embeddings.

Initially, the input waveform, represented as a 1-D float32 Tensor or NumPy array, undergoes internal preprocessing within the YAMNet model. This involves computing a spectrogram using the Short-Time Fourier Transform with a window size of 25 ms and a hop size of 10 ms, followed by mapping to 64 mel bins covering a frequency range of 125-7500 Hz. Subsequently, a stabilized log mel spectrogram is computed, addressing potential issues with zero logarithms. These features are then segmented into 50\%-overlapping examples of 0.96 seconds, encompassing 64 mel bands and 96 frames of 10 ms each.

The resulting 96x64 patches are fed into the YAMNet architecture model, culminating in a 3x2 array of activations for 1024 kernels at the top of the convolution. These activations are averaged to derive a 1024-dimensional embedding, with the YAMNet architecture comprising 14 convolutional layers of varying sizes. Following this, a Global Average Pooling (2D) step is employed to obtain the final embeddings.

Given that each audio snippet spans 10 seconds, 20 embeddings of vector size 1024 are produced for each chunk. These embeddings are concatenated, yielding a vector size of 1024*20 for each chunk. Subsequently, this concatenated data is fed into a neural network model, starting with an input layer expecting data with 1024 features. The subsequent dense layers, each followed by batch normalization and dropout (0.3), progressively reduce dimensionality. The output layer, employing softmax activation, produces a probability distribution over the 4 classes defined in the class names. Finally, the model is compiled using the Adam optimizer with an adaptable learning rate, and trained using the CategoricalCrossentropy loss function, while evaluating performance using metrics such as accuracy and AUC.

## Results 

### Training and Validation Metrics

Below are the training and validation metrics for both binary and multiclass classification models:


| Metric                        | Value  |
|-------------------------------|--------|
| Binary Training Accuracy      | 0.8645 |
| Binary Validation Accuracy    | 0.8618 |
| Binary Train Loss             | 0.2956 |
| Binary Validation Loss        | 0.3052 |
| Binary Train AUC              | 0.9750 |
| Binary Validation AUC         | 0.9736 |
| Multiclass Training Accuracy  | 0.6519 |
| Multiclass Validation Accuracy| 0.6645 |
| Multiclass Train Loss         | 0.5262 |
| Multiclass Validation Loss    | 0.5722 |

*Table 2: Training and Validation Metrics for Binary and Multiclass Classification*


