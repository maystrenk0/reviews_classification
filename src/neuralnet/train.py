import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, columns_idx, criterion, optimizer, epochs=10):
    # Empty lists to track metrics
    epoch_count, train_loss_values, valid_loss_values, valid_metric_values = [], [], [], []

    # Early stopping parameters
    patience = int(epochs)
    best_val_loss = float("inf")
    best_epoch = -1
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        model.sentiment_model.train()
        running_loss = 0.0  # Initialize running loss for each epoch

        for i, (inputs, labels) in enumerate(train_loader):
            input_ids = inputs[:, columns_idx["INPUT_IDS_START"] : columns_idx["INPUT_IDS_END"]]
            attention_mask = inputs[
                :, columns_idx["ATTENTION_MASK_START"] : columns_idx["ATTENTION_MASK_END"]
            ]
            x_cat = inputs[:, columns_idx["CATEGORICAL_START"] : columns_idx["CATEGORICAL_END"]]
            x_cont = inputs[:, columns_idx["NUMERIC_START"] : columns_idx["NUMERIC_END"]]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask, x_cat, x_cont).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate running loss
            running_loss += loss.item()

        # Validation phase
        model.eval()
        model.sentiment_model.eval()
        val_loss = 0.0
        all_labels, all_preds = [], []  # To store labels and predictions for ROC-AUC

        with torch.no_grad():
            for inputs, labels in val_loader:
                input_ids = inputs[
                    :, columns_idx["INPUT_IDS_START"] : columns_idx["INPUT_IDS_END"]
                ]
                attention_mask = inputs[
                    :, columns_idx["ATTENTION_MASK_START"] : columns_idx["ATTENTION_MASK_END"]
                ]
                x_cat = inputs[
                    :, columns_idx["CATEGORICAL_START"] : columns_idx["CATEGORICAL_END"]
                ]
                x_cont = inputs[:, columns_idx["NUMERIC_START"] : columns_idx["NUMERIC_END"]]

                outputs = model(input_ids, attention_mask, x_cat, x_cont).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Store labels and outputs for ROC-AUC calculation
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().numpy())

        # Average the losses over the dataset sizes
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Calculate ROC-AUC for the validation set
        val_metric = roc_auc_score(all_labels, all_preds)

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            early_stop_counter = 0  # Reset counter if validation loss improves
        else:
            early_stop_counter += 1  # Increment counter if no improvement

        if early_stop_counter >= patience:
            print("Early stopping triggered!")
            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val ROC-AUC: {val_metric:.4f}"
            )
            break

        # Store epoch metrics
        if (epochs - epoch - 1) % int(epochs / 25) == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val ROC-AUC: {val_metric:.4f}"
            )
            epoch_count.append(epoch)
            train_loss_values.append(avg_train_loss)
            valid_loss_values.append(avg_val_loss)
            valid_metric_values.append(val_metric)

    print("BEST EPOCH:", best_epoch)
    print("BEST LOSS:", best_val_loss)

    # Plot Loss Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_count, train_loss_values, label="Training Loss")
    plt.plot(epoch_count, valid_loss_values, label="Validation Loss")
    plt.title("Training & Validation Loss Curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot ROC-AUC Curve
    plt.subplot(1, 2, 2)
    plt.plot(epoch_count, valid_metric_values, label="Validation ROC-AUC")
    plt.title("Validation ROC-AUC Curve")
    plt.ylabel("ROC-AUC")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()
