import torch
import numpy as np
from copy import deepcopy

from sklearn import metrics
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, epochs, batch_size_train,
                 batch_size_val, batch_size_test, hidden_size, lambda_p, n_attention_heads, device):
        """
        Initialize the Trainer object with all the necessary parameters.

        Args:
            model: The model to be trained.
            train_loader: The data loader for the training data.
            val_loader: The data loader for the validation data.
            test_loader: The data loader for the test data.
            optimizer: The optimizer for training the model.
            epochs: The number of epochs for training.
            batch_size_train: The batch size for training.
            batch_size_val: The batch size for validation.
            batch_size_test: The batch size for testing.
            hidden_size: The size of the hidden layers.
            lambda_p: Regularization factor for the attention loss term.
            n_attention_heads: Number of attention heads in the model.
            device: The device to train the model on (CPU/GPU).
        """
        self.model = model.to(device)  # Move the model to the specified device (e.g., GPU or CPU)

        # Store the data loaders, optimizer, and other configuration parameters
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.hidden_size = hidden_size
        self.lambda_p = lambda_p
        self.n_attention_heads = n_attention_heads
        self.best_val_acc_sequence = -1000  # Initialize the best validation accuracy for sequence classification
        self.best_val_acc_entry = -1000  # Initialize the best validation accuracy for key label classification
        self.best_val_model = None  # Will store the best model based on validation performance
        self.device = device

    def train(self):
        """Train the model using triplet loss and optimize based on validation accuracy."""
        for epoch in range(self.epochs):
            self.model.train()  # Set the model to training mode
            epoch_loss = []  # Store the loss values for this epoch

            # Iterate through the training data
            for sequence, sequence_label, _, semi in self.train_loader:
                # Move the input data to the device (e.g., GPU or CPU)
                sequence, sequence_label, semi = sequence.to(self.device), sequence_label.to(self.device), semi.to(
                    self.device)

                self.optimizer.zero_grad()  # Reset gradients before backpropagation

                # Forward pass through the model to compute the triplet loss and other outputs
                triplet_loss, _, _, M, _ = self.model(sequence, sequence_label, semi, self.batch_size_train,
                                                      self.hidden_size)

                # Regularization term: compute attention weights and use them for regularization
                I = torch.eye(self.n_attention_heads * 2).to(self.device)
                c_na = torch.cat((self.model.c_n, self.model.c_a), 1)
                CCT = c_na @ c_na.transpose(1, 2)
                P = torch.mean((CCT.squeeze() - I) ** 2)

                # Total loss: sum of the triplet loss and the regularization term
                loss = triplet_loss + self.lambda_p * P
                loss.backward()  # Backpropagate the loss to compute gradients
                self.optimizer.step()  # Update model parameters using the gradients
                epoch_loss.append(loss.item())  # Append the loss for this batch to the epoch loss list

            # After each epoch, evaluate the model's performance on the validation set
            self._evaluate(epoch, epoch_loss)

    def _evaluate(self, epoch, epoch_loss):
        """Evaluate model performance on validation data and update best model if necessary."""
        self.model.eval()  # Set the model to evaluation mode
        correct_sequence, correct_entry = 0, 0  # Initialize counters for correct predictions

        with torch.no_grad():  # Disable gradient computation during evaluation
            # Iterate over the validation dataset
            for sequence, sequence_label, key_label, semi in self.val_loader:
                sequence_label, key_label = sequence_label.to(self.device), key_label.to(self.device)
                hidden = self.model.embedding(sequence.to(self.device))  # Compute hidden representations
                M, A = self.model.self_attention(hidden)  # Apply the self-attention mechanism to the hidden states

                # Compute cosine distances between the sequence and the normal/abnormal class representations
                n_dists = 0.5 * (1 - self.model.cosine_dist(M,
                                                            torch.repeat_interleave(self.model.c_n, self.batch_size_val,
                                                                                    dim=0)))
                a_dists = 0.5 * (1 - self.model.cosine_dist(M,
                                                            torch.repeat_interleave(self.model.c_a, self.batch_size_val,
                                                                                    dim=0)))

                # Compute scores for normal and abnormal sequences by averaging the cosine distances
                n_scores, a_scores = torch.mean(n_dists, dim=1), torch.mean(a_dists, dim=1)

                # Assign labels based on which score (normal or abnormal) is smaller
                pred_label_batch = torch.where(n_scores < a_scores, 0, 1)

                # Determine the best attention heads based on the distances
                _, n_best_heads = torch.min(n_dists, dim=1)
                _, a_best_heads = torch.min(a_dists, dim=1)
                best_att_heads = torch.where(pred_label_batch == 0, n_best_heads, a_best_heads)
                best_head_l = best_att_heads.tolist()

                # Generate predicted key labels based on the attention heads
                pred_key_label_l = [A[t, best_head_l[t], :].tolist() for t in range(len(sequence_label))]
                pred_key_label_t = torch.tensor(pred_key_label_l).to(self.device)

                # Create index masks for normal (index0) and abnormal (index1) predictions
                index0, index1 = pred_label_batch == 0, pred_label_batch == 1

                # Set the key label values to 0 for normal predictions
                pred_key_label_t[index0, :] = 0
                # For abnormal predictions, threshold the key label values based on the threshold (0.01)
                pred_key_label_t[index1] = torch.where(pred_key_label_t[index1] > 0.01, 1.0, 0.0)

                # Count correct sequence predictions (i.e., correct class assignments)
                correct_sequence += (pred_label_batch == sequence_label).sum().item()
                # Count correct key label predictions
                correct_entry += (
                        torch.reshape(pred_key_label_t, (-1,)) == torch.reshape(key_label, (-1,))).sum().item()

        # If the model improves on both sequence and entry classification accuracy, update the best model
        if correct_sequence > self.best_val_acc_sequence and correct_entry > self.best_val_acc_entry:
            self.best_val_acc_sequence, self.best_val_acc_entry = correct_sequence, correct_entry
            self.best_val_model = deepcopy(self.model.state_dict())  # Store the best model's state

        # Print the average loss for the current epoch
        print(f'Epoch {epoch:02d}: {np.mean(epoch_loss)}')

    def test(self, output_dir, n_attention_heads, batch_size_test):
        """Perform evaluation on test data after training and log results."""
        # Load the best model from validation
        self.model.load_state_dict(self.best_val_model)
        self.model.eval()  # Set model to evaluation mode

        pred_seq_label, true_seq_label = [], []  # Store predictions and true sequence labels
        pred_key_label, true_key_label = [], []  # Store predictions and true key labels

        top_entry = [[] for x in range(n_attention_heads)]

        with torch.no_grad():  # Disable gradient computation during testing
            # Iterate over the test dataset
            for sequence, sequence_label, key_label, _ in tqdm(self.test_loader):
                pred_key_label_l = []
                true_key_label += torch.reshape(key_label, (-1,)).tolist()  # Append true key labels
                true_seq_label += sequence_label.tolist()  # Append true sequence labels

                hidden = self.model.embedding(sequence.to(self.device))  # Get hidden representations
                M, A = self.model.self_attention(hidden)  # Apply self-attention mechanism

                # Compute cosine distances for normal and abnormal sequences
                n_dists = 0.5 * (1 - self.model.cosine_dist(M, torch.repeat_interleave(self.model.c_n,
                                                                                       self.batch_size_test, dim=0)))
                a_dists = 0.5 * (1 - self.model.cosine_dist(M, torch.repeat_interleave(self.model.c_a,
                                                                                       self.batch_size_test, dim=0)))

                # Compute scores for normal and abnormal sequences
                n_scores, a_scores = torch.mean(n_dists, dim=1), torch.mean(a_dists, dim=1)

                # Assign labels based on which score (normal or abnormal) is smaller
                pred_label_batch = torch.where(n_scores < a_scores, 0, 1)
                pred_seq_label += pred_label_batch.tolist()

                # Determine the best attention heads based on the distances
                _, n_best_heads = torch.min(n_dists, dim=1)
                _, a_best_heads = torch.min(a_dists, dim=1)
                best_att_heads = torch.where(pred_label_batch == 0, n_best_heads, a_best_heads)
                best_head_l = best_att_heads.tolist()

                index0, index1 = pred_label_batch == 0, pred_label_batch == 1

                # Generate predicted key labels based on the attention heads
                for t in range(len(sequence_label)):
                    pred_key_label_l.append(A[t, best_head_l[t], :].tolist())

                pred_key_label_t = torch.tensor(pred_key_label_l).to(self.device)
                pred_key_label_t[index0, :] = 0
                pred_key_label_t[index1] = torch.where(pred_key_label_t[index1] > 0.01, 1.0, 0.0)
                pred_key_label += list(map(int, torch.reshape(pred_key_label_t, (-1,)).tolist()))

                for i in range(batch_size_test):
                    top_entry[best_head_l[i]] += np.array(sequence[i])[pred_key_label_t.cpu().numpy()[i] == 1].tolist()

        # Print evaluation metrics (classification report, confusion matrix, ROC AUC score)
        print(metrics.classification_report(true_key_label, pred_key_label, digits=4))
        print(metrics.confusion_matrix(true_key_label, pred_key_label))

        fpr, tpr, _ = metrics.roc_curve(true_key_label, pred_key_label, pos_label=1)
        auc_score = metrics.auc(fpr, tpr)
        print(auc_score)

        # Save the evaluation results to a file
        with open(output_dir + 'result.txt', 'w') as f:
            f.write('Entry anomaly detection on detected sequences:' + '\n')
            f.write(str(metrics.classification_report(true_key_label, pred_key_label, digits=4)) + '\n')
            f.write(str(metrics.confusion_matrix(true_key_label, pred_key_label)) + '\n')
            f.write(str(auc_score) + '\n')
            f.write('-' * 50 + '\n')

        return top_entry
