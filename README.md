# Email Spam Detection using BERT

This project focuses on detecting spam emails using a BERT (Bidirectional Encoder Representations from Transformers) model. The primary goal is to build a spam classifier capable of distinguishing spam emails from legitimate ones. The model is fine-tuned using the Enron-Spam dataset and deployed using Docker, FastAPI, and Huggingface Spaces.

## Table of Contents
- [Introduction](#introduction)
- [Why BERT?](#why-bert)
- [Data Set and Processing](#data-set-and-processing)
- [Training and Testing Data Split](#training-and-testing-data-split)
- [Fine-tuning the Model](#fine-tuning-the-model)
- [Model Evaluation](#model-evaluation)
- [Model Saving](#model-saving)
- [Memory Usage](#memory-usage)
- [Model Deployment](#model-deployment)
  - [Step-by-step Deployment Process](#step-by-step-deployment-process)
- [Inference](#inference)
- [Conclusion](#conclusion)

---

## Introduction
This project utilizes BERT, a transformer-based model, to detect spam emails. The model is trained and fine-tuned using the Enron-Spam dataset, which contains both spam and non-spam emails. We use tools like Pandas, PyTorch, TensorFlow, and Huggingface's `Transformers` library for model building, and FastAPI for serving the model in production.

## Why BERT?
BERT is a state-of-the-art transformer-based model that processes text bidirectionally, allowing it to understand the context of words within sentences. This is particularly useful for spam detection, where understanding the meaning of an email is crucial for classification. BERT is pre-trained on large corpora, making it highly accurate for NLP tasks, including spam detection.

## Data Set and Processing
The Enron-Spam dataset is used, consisting of thousands of emails categorized as spam or ham (non-spam). Data extraction and processing involved the following steps:

1. **Data Extraction**: Extracted raw text from `.txt` files and saved them into a `.csv` format using Pandas.
2. **Class Imbalance**: The original dataset had 4500 spam emails and 1500 ham emails. The spam emails were downsampled to 1500 to create a balanced dataset.
3. **Text Preprocessing**: 
    - Removal of URLs, HTML tags, non-alphanumeric characters, and extra spaces.
    - Stopword removal and lemmatization were not necessary, as BERT understands context well.

## Training and Testing Data Split
The processed data was split into an 80:20 ratio for training and testing. This split ensures sufficient data for training while keeping enough samples to evaluate the model.

## Fine-tuning the Model
To fine-tune the BERT model:
1. **BERT Model**: Used the `bert-base-uncased` pre-trained model from Huggingface.
2. **Tokenization**: The dataset was tokenized using BERT's tokenizer.
3. **Fine-tuning**: The model was fine-tuned for 3 epochs, which was sufficient to achieve high accuracy (98%).
4. **Hardware**: Training on a CPU took 1.5 hours; using a GPU reduced the time to 3 minutes.

## Model Evaluation
The model achieved 98% accuracy on the testing set. Below is the model performance breakdown:
- **Confusion Matrix**: Shows how well the model classifies emails.
- **Classification Report**: Provides precision, recall, and F1-score for both spam and non-spam classifications.

## Model Saving
The final model is saved in a format compatible with the Huggingface `Transformers` library, ensuring that both the model and the tokenizer are saved separately for easier loading and deployment.

## Memory Usage
The BERT model used in this project has over 100 million parameters. The memory consumption is approximately:
- **Memory Consumption in MB**: 417.65 MB
- **Memory Consumption in GB**: 0.408 GB

## Model Deployment
The trained model was deployed using Docker, FastAPI, and Huggingface Spaces.

### Step-by-step Deployment Process
1. **Step 1: Dockerization**:
    - The model was containerized using Docker. This ensures that the model can run in any environment with the required dependencies.
    - The Docker image includes the environment setup (libraries such as Huggingface's `Transformers`, PyTorch, etc.) and the fine-tuned BERT model.

2. **Step 2: FastAPI Setup**:
    - A REST API was developed using FastAPI to serve the model for inference.
    - The API allows users to send POST requests with email text, which the model processes to classify as spam or non-spam.

3. **Step 3: Hosting on Huggingface Spaces**:
    - The project was pushed to a Git repository and linked with Huggingface Spaces.
    - Huggingface Spaces builds and hosts the Dockerized model, providing a publicly accessible API endpoint.

## Inference
Once deployed, the model can be accessed through the Huggingface Spaces API. To make predictions:
1. Send a POST request with the email text to the API endpoint.
2. The model will return whether the email is classified as spam or not.

Example API request:
```bash
curl -X POST "https://api.huggingface.co/spam-detection" -H "Content-Type: application/json" -d '{"email": "You have won $1000!"}'
```

## Conclusion

This project successfully demonstrates the complete pipeline for detecting spam emails using a fine-tuned BERT model. The model was trained, fine-tuned, and deployed using modern tools like Docker, FastAPI, and Huggingface Spaces, providing a scalable and efficient solution for real-time spam detection.


## Contact
Feel free to reach out for any questions or collaborations!

- **Email**: [satyapanthi.t@northeastern.edu](mailto:satyapanthi.t@northeastern.edu)
- **LinkedIn**: [@tapaswi-v-s](https://www.linkedin.com/in/tapaswi-v-s/)