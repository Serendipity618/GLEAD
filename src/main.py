import torch
import argparse
from torch import optim

from model import GLEAD
from preprocessing import LogPreprocessor, DataProcessor
from dataloader import DataLoaderWrapper
from trainer import Trainer
from analyzer import LogAnalyzer  # Assuming LogAnalyzer is in a file named 'log_analyzer.py'

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate GLEAD model for anomaly detection.')
    parser.add_argument('--output_path', type=str, default='./output/result/', help='Path to save evaluation results')
    parser.add_argument('--top_log_path', type=str, default='./output/top_log/', help='Path to save top log entries')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lambda_p', type=float, default=1.0, help='Regularization parameter')
    parser.add_argument('--hidden_size', type=int, default=150, help='Hidden layer size')
    parser.add_argument('--attention_size', type=int, default=300, help='Attention size')
    parser.add_argument('--n_attention_heads', type=int, default=5, help='Number of attention heads')
    parser.add_argument('--batch_size_train', type=int, default=60, help='Batch size for training')
    parser.add_argument('--batch_size_val', type=int, default=20, help='Batch size for validation')
    parser.add_argument('--batch_size_test', type=int, default=1000, help='Batch size for testing')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for optimizer')
    parser.add_argument('--dataset_path', type=str, default='data/BGL.log_structured_v1.csv', help='Path to dataset')
    args = parser.parse_args()

    # Data Preprocessing
    print("Loading and preprocessing data...")
    dataset = LogPreprocessor(args.dataset_path).slide_window()
    processor = DataProcessor(dataset)
    batch_sizes = {'train': args.batch_size_train, 'val': args.batch_size_val, 'test': 1000}
    data_loader = DataLoaderWrapper(processor, batch_sizes)

    # Initialize Model
    print("Initializing model...")
    model = GLEAD(args.attention_size, args.n_attention_heads, args.hidden_size, processor.logkeys, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training and Evaluation
    print("Starting training...")
    trainer = Trainer(model, data_loader.train_loader, data_loader.val_loader, data_loader.test_loader, optimizer,
                      args.epochs, args.batch_size_train, args.batch_size_val, args.batch_size_test, args.hidden_size,
                      args.lambda_p, args.n_attention_heads, device)
    trainer.train()
    print("Training completed. Evaluating model...")
    top_entry = trainer.test(args.output_path, args.n_attention_heads, args.batch_size_test)

    print("Model evaluation complete.")

    # Extract and save top log entries for each attention head after training and evaluation
    print("Extracting top log entries for each attention head...")
    LogAnalyzer.save_top_entries(top_entry, args.n_attention_heads, args.top_log_path + 'top_entry.txt')
    print("Top log entries saved.")


if __name__ == "__main__":
    main()
