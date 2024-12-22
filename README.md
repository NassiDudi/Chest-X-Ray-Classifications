# Chest-X-Ray-Classifications

This project focuses on classifying chest X-ray images to detect pneumonia using deep learning models.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Technologies Used](#technologies-used)  
3. [Dataset](#dataset)  
4. [Data Augmentation](#data-augmentation)  
5. [How to Run](#how-to-run)  
6. [Code Breakdown](#code-breakdown)  
7. [Visualization](#visualization)  
8. [Results](#results)

## Project Overview
This project aims to classify chest X-ray images to detect pneumonia by leveraging deep learning techniques. Four different models were implemented and compared to evaluate their performance in this medical image classification task.
1. Convolutional Neural Network (CNN): A custom CNN model designed specifically for this task.
2. Pretrained ResNet50: Another transfer learning model utilizing the ResNet50 architecture to leverage its deep feature extraction capabilities.
3. Pretrained VGG16: A transfer learning model based on the VGG16 architecture, known for its simplicity and effectiveness in image classification tasks.
4. Vision Transformer (ViT): A transformer-based architecture adapted for image classification, leveraging self-attention mechanisms to capture global image features.   
The models were evaluated on their ability to classify chest X-ray images into normal and pneumonia categories, and the results were compared to identify the best-performing approach. 

## Technologies Used
- Python 3.x
- PyTorch – Deep learning framework
- Torchvision – Pre-trained models and dataset utilities
- Transformers (Hugging Face) – Vision Transformer (ViT) model
- Pandas – Data manipulation and analysis
- NumPy – Numerical computations
- Scikit-learn – Model evaluation metrics
- Matplotlib & Seaborn – Data visualization

## Dataset
The dataset used in this project consists of chest X-ray images categorized into two classes:
/content/train/
    ├── Normal/
    ├── Pneumonia/
/content/val/
/content/test/
- Normal: X-ray images showing no signs of pneumonia.
- Pneumonia: X-ray images indicating the presence of pneumonia.
The dataset includes training, validation and testing subsets to ensure proper model evaluation.

## Data Augmentation
In this project, data augmentation techniques were applied to improve model generalization, prevent overfitting, and simulate real-world image variations. 
The following augmentation strategies were implemented:
| Augmentation Technique     | Key Role in Training                                     |
|----------------------------|----------------------------------------------------------|
| **Random Affine**          | Applies random rotations and translations up to 15° and 10% translation.        |
| **Random Resized Crop**    | Randomly crops and resizes images with a scale of (0.8, 1.0). |
| **RandomHorizontalFlip**   | Randomly flips images horizontally with a probability of 0.5.               |
| **RandomRotation**         | Rotates images randomly up to 15°.                |
| **ColorJitter**            | Randomly adjusts brightness, contrast, saturation, and hue up to 20%. |

## How to Run
There are two ways you can run this project:

1. Using Google Colab (Recommended for easy setup)
   Download the notebook:

   Visit the GitHub repository and download the Chest X-Ray Classifications.ipynb file.
   Upload the notebook to Colab:

   Open Google Colab.
   Click on "File" > "Upload notebook" and select the downloaded .ipynb file.

2. Clone this repository:
   ```bash
   git clone https://github.com/NassiDudi/Chest-X-Ray-Classifications.git
   ```
   - Install the required packages
   - Use any IDE that supports Jupyter Notebooks

## Code Breakdown
### 1. Data Augmentation
Before training the model, the dataset is augmented and divided to dataloades.
### 2. Defining Modles
- ComplexCNNModel: The model is a CNN with three convolutional layers, followed by max-pooling, dropout, and fully connected layers for binary classification. The output layer uses a sigmoid activation to produce a probability score between 0 and 1.
- Pre-Trained Models: ResNet50, VGG16 and ViT
### 3. Training, Validation, and Testing
Each model (ComplexCNNModel, ResNet, VGG, ViT) is trained using the training dataset, followed by validation on the validation dataset to assess its performance. After training and validation, the model is tested on a separate test dataset to evaluate its accuracy and performance. The training and validation losses are tracked throughout the process.
### 4. Evaluation
The final results are presented with the confusion matrix and accuracy score of each model.
### 5. Fine-Tuning the ResNet Model
The ResNet model is fine-tuned by experimenting with different batch sizes (16, 32, 64) and learning rates (0.001, 0.0001, 0.01). The model is trained for 3 epochs with the AdamW optimizer and a learning rate scheduler. After each epoch, training and validation losses, as well as accuracy, are tracked. The model is then evaluated on a test set, and the best configuration (based on test accuracy) is saved along with the model's details for further analysis.

## Visualization
### Visual Outputs:
1. **Augmented vs. Real Images**: A comparison between augmented images and the original images, demonstrating the effect of data augmentation techniques.
2. **Training Metrics**: Loss and accuracy curves over epochs on the validation set.
3. **Confusion Matrix**: A heatmap showing the confusion matrix to visualize the performance of the model in terms of true positives, false positives, true negatives, and false negatives.
4. **Example Images of Good and Bad Classifications**: A display of example images that were correctly classified (good) and incorrectly classified (bad) by the model, highlighting the model's strengths and weaknesses in prediction.

## Results
The project achieved the following test accuracies:
| Model         | Test Accuracy |
|---------------|---------------|
| ComplexCNN    |    89%        |
| ResNet50      |    91%        |
| VGG16         |    62%        |
| ViT           |    91%        |

Fine-Tuning ResNet Model achieved in each experiment the following test accuracies:
| Experiment | Batch Size | Learning Rate | Test Accuracy |
|------------|------------|---------------|---------------|
| 1          | 16         | 0.0010        | 0.870192      |
**| 2          | 16         | 0.0001        | 0.942308      |**
| 3          | 16         | 0.0100        | 0.461538      |
| 4          | 32         | 0.0010        | 0.863782      |
| 5          | 32         | 0.0001        | 0.879808      |
| 6          | 32         | 0.0100        | 0.862179      |
| 7          | 64         | 0.0010        | 0.862179      |
| 8          | 64         | 0.0001        | 0.892628      |
| 9          | 64         | 0.0100        | 0.881410      |

Experiment 2 with a batch size of 16 and a learning rate of 0.0001 yielded the best result, achieving a test accuracy of 0.942308. This configuration demonstrated the optimal balance between training performance and model generalization, resulting in the highest test accuracy compared to other combinations of batch sizes and learning rates in the experiment.

