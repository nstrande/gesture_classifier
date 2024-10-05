# Hand Gesture Recognition

This project implements a comprehensive hand gesture recognition system using Python, OpenCV, MediaPipe, and PyTorch. The system includes data collection, annotation, model training, and real-time recognition of hand gestures via webcam. It has been developed and tested on Macs with M2 chips.

https://github.com/user-attachments/assets/f08fe363-cc4f-4f14-84ec-669c04f9d417

## Features

- Data collection and annotation of hand gestures
- Training of a custom PyTorch model for gesture recognition
- Real-time hand detection using MediaPipe
- Gesture recognition using the trained model
- Visualization of hand landmarks and gesture predictions
- Optimized for Macs with M2 chips

## Prerequisites

Before you begin, ensure you have met the following requirements:

- macOS running on an M2 chip (or compatible Apple Silicon)
- Python 3.10+ (preferably installed via Conda)
- OpenCV
- MediaPipe
- PyTorch (version compatible with Apple Silicon)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/hand-gesture-recognition.git
   cd gesture_classifier
   ```

2. Set up a and activate virtual environment (recommended)

3. Install the required packages:
   ```
   make install_dev
   ```

   Note: Ensure that you're installing versions of the packages that are compatible with Apple Silicon. For PyTorch, you may need to install it separately following the instructions on the official PyTorch website for Mac M1/M2.

## Usage

The system consists of several steps:

### 1. Data Collection and Annotation

To collect and annotate data for training:

```
make run_annotator
```

This script will access your Mac's webcam and guide you through the process of recording different hand gestures. Follow the on-screen instructions to annotate each gesture.

### 2. Data Processing

After collecting and annotating data, process the raw data into training data:

```
make preprocess_data
```


### 2. Model Training

After processing raw data, train the model with:

```
python train.py
```

This script will train a PyTorch model using the collected dataset and save the trained model in the `models/` directory. The training process is optimized for M2 chip performance.

### 3. Real-time Gesture Recognition

To run real-time hand gesture recognition using the trained model:

```
make run_main
```

- The program will access your Mac's webcam and open a window showing the video feed.
- Hold your hand in front of the camera. The program will detect your hand, draw landmarks, and display the predicted gesture.
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
    └── utils
```

## Performance Considerations for M2 Macs

- The application leverages the M2 chip's Neural Engine for improved machine learning performance.
- GPU acceleration is utilized where possible to enhance real-time processing capabilities.
- Ensure that your PyTorch installation is optimized for Apple Silicon to get the best performance.

## Customization

- Use `make run_annotator` to change the number of gestures and dataset size.
- Modify the model architecture in `src/models/moden_config/classification_nn.py` to experiment with different network structures.
- Adjust the `Hands()` parameters in the `initialize_mediapipe()` function in `src/main.py` to change hand detection sensitivity.

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

- [MediaPipe](https://mediapipe.dev/) for the hand detection framework.
- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [OpenCV](https://opencv.org/) for image processing capabilities.
- Apple for M2 chip optimizations in machine learning frameworks.
