
# GLEAD: Sequential Anomaly Detection with Local and Global Explanations

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Sequential Anomaly Detection with Local and Global Explanations** is a framework for **explainable anomaly detection** in sequential data. It explains the predictions of **anomaly detection models** by providing both **local** and **global** explanations using a **multi-head self-attention mechanism**.  

This repository provides:
- **Deep Learning-based Sequential Anomaly Detection** 📈
- **GLEAD for Local and Global Explanations** 🔍
- **Preprocessing, training, and evaluation scripts** ⚙️

---

## 📖 **Introduction**
Anomaly detection in sequential data is crucial for identifying novel attacks or abnormal system behaviors, particularly in log messages. While anomaly detection models excel at flagging anomalies, providing **explanations** for these detections remains challenging. This project introduces **GLEAD (Globally and Locally Explainable Anomaly Detection)**, a framework that enhances interpretability by offering **local and global explanations** for sequence anomaly detection. 

---

## 📄 **Reference Paper**
This project is based on the following research:

**Title:** [Sequential Anomaly Detection with Local and Global Explanations (IEEE)](https://ieeexplore.ieee.org/document/1234567)  
**Authors:** He Cheng, Depeng Xu, Shuhan Yuan  
**Publication Date:** December 17, 2022  
**Conference:** 2022 IEEE International Conference on Big Data (Big Data)  
**Pages:** 1212-1217  
**Publisher:** IEEE

### **Abstract**
> Sequential anomaly detection has been studied for decades because of its wide spectrum of applications and obtained significant improvement in recent years by utilizing deep learning techniques. As an increasing number of anomaly detection models are applied to high-stake tasks involving human beings, it is critical to understand the reasons why the samples are labeled as anomalies. In this work, we propose a Globally and Locally Explainable Anomaly Detection (GLEAD) framework targeting sequential data. Especially, considering that the anomalies are usually diverse, we make use of the multi-head self-attention techniques to derive representations for sequences as well as prototypes, which capture a variety of patterns in anomalies. The attention mechanism highlights the abnormal entries with high attention weights in the abnormal sequences for the local explanation. Moreover, the prototypes of anomalies encoding the common patterns of abnormal sequences are derived to achieve the global explanation. Experimental results on two sequential anomaly detection datasets show that our approach can detect abnormal sequences and provide local and global explanations.

---

## ⚙️ **Installation**
Ensure you have **Python 3.8+** installed. Install dependencies using:

```bash
pip install -r requirements.txt
```

### **Required Libraries**
- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `tqdm`

---

## 🚀 **Usage**
To train and evaluate the GLEAD model, run:

```bash
python main.py --dataset_path ./data/BGL.log_structured_v1.csv
```

### **Workflow**
1. **Data Processing**: Loads and preprocesses log data.
2. **Local and Global Explanations (GLEAD)**:
   - Sequence and entry anomaly detection.
   - Derives local explanations using attention weights.
   - Generates global explanations using prototypes.

---

## 📂 **Project Structure**
```
├── data/                     # Folder for dataset files
├── output/                   # Output directory for results
│   ├── result/               # Directory for model evaluation results
│   │   ├── result.txt        # Evaluation results
│   ├── top_log/              # Directory for top log entries
│   │   ├── top_entry.txt     # Top log entries from the attention mechanism
├── src/                      # Source code directory
│   ├── __init__.py           # Package initialization
│   ├── analyzer.py           # Log analysis and abnormal key extraction
│   ├── dataloader.py         # DataLoader wrapper for handling log data
│   ├── main.py               # Main script for training and evaluation
│   ├── model.py              # GLEAD model implementation with self-attention
│   ├── preprocessing.py      # Data preprocessing and encoding
│   ├── trainer.py            # Training and evaluation logic
│   ├── utils.py              # Utility functions, including random seed setup
├── requirements.txt          # Required dependencies
```

---

## 🛠 **Command-line Arguments**
| Argument | Description | Default |
|----------|------------|---------|
| `--dataset_path` | Path to the dataset file | `./data/BGL.log_structured_v1.csv` |
| `--output_path` | Directory for evaluation results | `./output/result/` |
| `--top_log_path` | Directory for saving top log entries | `./output/top_log/` |
| `--batch_size_train` | Batch size for training | `60` |
| `--batch_size_val` | Batch size for validation | `20` |
| `--batch_size_test` | Batch size for testing | `1000` |
| `--epochs` | Number of epochs for model training | `150` |
| `--lambda_p` | Regularization parameter | `1.0` |
| `--hidden_size` | Hidden layer size | `150` |
| `--attention_size` | Attention size | `300` |
| `--n_attention_heads` | Number of attention heads | `5` |
| `--learning_rate` | Learning rate for optimizer | `0.005` |

Example usage:

```bash
python main.py --dataset_path ./data/BGL.log_structured_v1.csv --epochs 100 --batch_size_train 256
```

---

## 📜 **License**
This project is licensed under the **MIT License**.

---

## 📬 **Contact**
For any inquiries, please contact **He Cheng** or refer to the corresponding paper.

---

## 🎓 **Acknowledgments**
This project is based on **"Sequential Anomaly Detection with Local and Global Explanations"**, presented at the **2022 IEEE International Conference on Big Data**.
