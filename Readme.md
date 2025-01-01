# Hand Gesture Recognition

This project implements a comprehensive hand gesture recognition system using Python, OpenCV, MediaPipe, and PyTorch. The system includes data collection, annotation, model training, and real-time recognition of hand gestures via webcam. It has been developed and tested on Macs with M2 chips.

https://github.com/user-attachments/assets/f08fe363-cc4f-4f14-84ec-669c04f9d417

## Features

- Advanced LSTM-based neural network architecture for dynamic gesture recognition
- Temporal sequence modeling enabling recognition of complex, time-dependent hand gestures
- Robust handling of varying gesture speeds and styles through LSTM's temporal memory capabilities
- Data collection and annotation of hand gestures
- Training of a custom PyTorch LSTM model for gesture recognition
- Real-time hand detection using MediaPipe
- Gesture recognition using the trained model
- Visualization of hand landmarks and gesture predictions
- Optimized for Macs with M2 chips

## Prerequisites

Before you begin, ensure you have met the following requirements:

- macOS running on an M2 chip (or compatible Apple Silicon)
- Python 3.10+ (preferably installed via Conda)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/nstrande/gesture_classifier
   cd gesture_classifier
   ```

2. Set up a and activate virtual environment (recommended)

3. Install the required packages:
   ```
   make install_dev
   ```

   Note: Ensure that you're installing versions of the packages that are compatible with Apple Silicon.

## Usage

The system consists of several steps:

### 1. Data Collection and Annotation

To collect and annotate data for training:

```
make generate_annotations
```

This script will access your Mac's webcam and guide you through the process of recording different hand gestures. Follow the on-screen instructions to annotate each gesture. The system captures temporal sequences of hand positions to enable dynamic gesture recognition.

### 2. Data Processing

After collecting and annotating data, process the raw data into training sequences:

```
make preprocess_data
```

The preprocessing step organizes the data into temporal sequences suitable for LSTM training, ensuring that the dynamic nature of gestures is preserved.

### 2. Model Training

After processing raw data, train the LSTM model with:

```
make train_model
```

This script will train a PyTorch LSTM model using the collected sequential dataset and save the trained model in the `models/` directory. The LSTM architecture is specifically designed to learn patterns in temporal hand movement sequences, allowing for recognition of dynamic gestures. The training process is optimized for M2 chip performance.

### 3. Real-time Gesture Recognition

To run real-time hand gesture recognition using the trained model:

```
make run_main
```

- The program will access your Mac's webcam and open a window showing the video feed.
- Hold your hand in front of the camera and perform gestures. The LSTM model will analyze the temporal sequence of hand positions to recognize dynamic gestures.
- The system maintains a temporal memory of hand movements, enabling accurate recognition of gestures performed at different speeds.
- Press 'q' or 'spacebar' to quit the program.

## Project Structure

```
├── assets
├── data
├── models
└── src
    ├── annotation_tools
    ├── data
    ├── models
    │   └── model_config
    ├── steps
    └── utils
```

## Model Architecture

The core of the system is built around an LSTM (Long Short-Term Memory) neural network architecture, which provides several key advantages for gesture recognition:

- Temporal Sequence Processing: The LSTM model processes sequences of hand positions over time, enabling recognition of dynamic gestures that involve movement patterns.
- Memory Capabilities: LSTM cells maintain both short-term and long-term memory of hand movements, allowing the model to recognize gestures regardless of their speed or slight variations in execution.
- Robust Recognition: The temporal nature of the LSTM architecture makes the system more robust to variations in how users perform gestures, as it focuses on the sequential patterns rather than static positions.

## Performance Considerations for M2 Macs

- The application leverages the M2 chip's Neural Engine for improved machine learning performance.
- GPU acceleration is utilized where possible to enhance real-time processing capabilities.
- The LSTM model's computations are optimized for Apple Silicon to ensure efficient processing of temporal sequences.
- Ensure that your PyTorch installation is optimized for Apple Silicon to get the best performance.

## Customization

- Use `make generate_annotations` to change the number of gestures and dataset size.
- Modify the LSTM architecture in `src/models/model_config/classification_lstm.py` to experiment with different network structures, such as:
  - Number of LSTM layers
  - Hidden state dimensions
  - Sequence length
  - Dropout rates for regularization
- Adjust the `init()` parameters in the `GestureRecognition()` function in `src/main.py` to change hand detection sensitivity.

## Troubleshooting

If you encounter any issues related to M2 compatibility:
- Ensure all dependencies are installed with Apple Silicon support.
- Check that you're using the correct Python interpreter (arm64 version).
- If using Rosetta 2 for any x86 applications, be aware that it may affect performance.

## Contributing

Contributions to this project are welcome. Please fork the repository and create a pull request with your changes. Ensure that any contributions maintain compatibility with M2 Macs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) for the hand detection framework.
- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [OpenCV](https://opencv.org/) for image processing capabilities.
- Apple for M2 chip optimizations in machine learning frameworks.