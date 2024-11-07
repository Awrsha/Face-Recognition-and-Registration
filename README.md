# Face Verification using FaceNet

![Face Verification Demo](https://via.placeholder.com/600x400.png?text=Face+Verification+Demo)

This project showcases an advanced face verification system built using the FaceNet deep learning architecture. The model is trained to accurately identify and verify individuals based on their facial features.

## Features

- **Robust Face Detection**: The system utilizes the MTCNN (Multi-Task Cascaded Convolutional Networks) algorithm to accurately detect and crop faces from input images.
- **Highly Accurate Face Verification**: The InceptionResNetV1 model, pre-trained on the VGGFace2 dataset, is fine-tuned to achieve state-of-the-art face verification performance.
- **Efficient Training Pipeline**: The training process includes data augmentation techniques, such as rotation, scaling, and translation, to enhance the model's robustness and generalization.
- **Customizable Settings**: The script allows users to adjust various hyperparameters, such as batch size, learning rate, and number of training epochs, to optimize the model's performance for their specific use case.
- **GPU Acceleration**: The model is designed to leverage GPU acceleration, providing faster inference and training times.
- **Detailed Visualization**: The training process is monitored using TensorBoard, which provides comprehensive visualizations of the model's performance, including loss, accuracy, and other relevant metrics.

## Getting Started

To get started with the Face Verification project, follow these steps:

1. **Clone the Repository**:
   ```
   git clone https://github.com/Awrsha/Face-Recognition-and-Registration.git
   cd Face-Recognition-and-Registration
   ```

2. **Set up the Environment**:
   - Install the required dependencies using `pip`:
     ```
     pip install -r requirements.txt
     ```
   - Ensure you have access to a compatible GPU (NVIDIA) and the necessary CUDA libraries installed.

3. **Prepare the Dataset**:
   - Download the LFW [Labeled Faces in the Wild](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) dataset and place it in the `lfw` directory.
   - Organize your own face data into a directory structure with each person's images in a separate subdirectory (e.g., `dataset/Person1`, `dataset/Person2`, etc.).

4. **Train the Model**:
   - Run the `training_script_v3.py` script to initiate the training process.
   - Monitor the training progress and evaluation metrics using TensorBoard:
     ```
     tensorboard --logdir=/path/to/Face-Recognition-and-Registration/logs
     ```

5. **Evaluate the Model**:
   - The trained model's checkpoint will be saved as `Face_Verification_v4.pth`.
   - You can use the saved model to perform face verification on new images or integrate it into your own application.

## Customization

The project's training script `training_script_v3.py` allows you to customize various aspects of the face verification system:

- **Data Augmentation**: Adjust the parameters of the `Data_and_Label_Augmentation` class to experiment with different data augmentation techniques.
- **Hyperparameters**: Modify the batch size, learning rate, number of training epochs, and other hyperparameters to optimize the model's performance.
- **Model Architecture**: You can explore different deep learning architectures, such as ResNet or VGGFace, by modifying the `InceptionResnetV1` model used in the project.
- **Visualization**: Customize the TensorBoard visualization by adding more metrics or adjusting the logging intervals.

## Resources

- **FaceNet Paper**: [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- **MTCNN Paper**: [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)
- **VGGFace2 Dataset**: [VGGFace2: A dataset for recognising faces across pose and age](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- **PyTorch FaceNet Implementation**: [PyTorch FaceNet Implementation](https://github.com/timesler/facenet-pytorch)

## Contributing

We welcome contributions to this project! If you have any ideas, bug fixes, or improvements, please feel free to submit a pull request. Let's collaborate to make this face verification system even better.

## License

This project is licensed under the [Apache-2.0 license](LICENSE).
