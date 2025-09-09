# train.py
import torch


def gat_train(gat_feature_extractor, train_loader_feature_extractor, test_loader_feature_extractor, criterion, optimizer, num_epochs, device, edge_index):

    for epoch in range(num_epochs):
        gat_feature_extractor.train()
        total_loss = 0
        # Autoencoder-style
        for features, labels in train_loader_feature_extractor:
            features, _ = features.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = gat_feature_extractor(features, edge_index)

            # Compute loss
            loss = criterion(outputs, features)
            total_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        # Validation
        gat_feature_extractor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in test_loader_feature_extractor:
                features, _ = features.to(device), labels.to(device)
                outputs = gat_feature_extractor(features, edge_index)
                loss = criterion(outputs, features)
                val_loss += loss.item()

        val_loss /= len(test_loader_feature_extractor)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {total_loss / len(train_loader_feature_extractor)}, Validation Loss: {val_loss}')

    # Save GAT model state
    torch.save(gat_feature_extractor.state_dict(), '../models/gat_feature_extractor.pth')


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, edge_index, train_dataset, scheduler):
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            output = model(batch_features, edge_index)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader)}")
        val_loss, all_aggregate, all_predictions, all_values = validate_model(model, val_loader, criterion, device,
                                                                              edge_index, train_dataset)
        # Update learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)  # Assume val_loss is the validation loss
        else:
            scheduler.step()

    return model, all_aggregate, all_predictions, all_values


def validate_model(model, val_loader, criterion, device, edge_index, train_dataset):
    # Collect predictions and ground-truth labels
    all_predictions = []
    all_aggregate = []
    all_values = []
    # Collect predictions and ground-truth labels
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch_features, batch_labels in val_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            predictions = model(batch_features, edge_index)

            original_features = train_dataset.get_feature_scaler().inverse_transform(batch_features.numpy())
            original_pred_values = train_dataset.get_label_scaler().inverse_transform(predictions.numpy())
            original_values = train_dataset.get_label_scaler().inverse_transform(batch_labels.numpy())
            all_aggregate.append(original_features)
            all_predictions.append(original_pred_values)
            all_values.append(original_values)

            loss = criterion(predictions, batch_labels)
            total_loss += loss.item()
        print(f"Validation Loss: {total_loss/len(val_loader)}")
    return total_loss, all_aggregate, all_predictions, all_values
