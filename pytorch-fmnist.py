import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import mlflow
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

CLASSIFIER = 'LogisticRegression'

if CLASSIFIER == 'LogisticRegression':
    experiment = mlflow.set_experiment("pytorch-fmnist-lr")
else:
    experiment = mlflow.set_experiment("pytorch-fmnist")

class RBM(nn.Module):
    def __init__(self, n_visible=784, n_hidden=256, k=1):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        # Initialize weights and biases
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.1)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.k = k  # CD-k steps

    def sample_h(self, v):
        # Given visible v, sample hidden h
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))  # p(h=1|v)
        h_sample = torch.bernoulli(p_h)                        # sample Bernoulli
        return p_h, h_sample

    def sample_v(self, h):
        # Given hidden h, sample visible v
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))  # p(v=1|h)
        v_sample = torch.bernoulli(p_v)
        return p_v, v_sample

    def forward(self, v):
        # Perform k steps of contrastive divergence starting from v
        v_k = v.clone()
        for _ in range(self.k):
            _, h_k = self.sample_h(v_k)    # sample hidden from current visible
            _, v_k = self.sample_v(h_k)    # sample visible from hidden
        return v_k  # k-step reconstructed visible

    def free_energy(self, v):
        # Compute the visible bias term for each sample in the batch
        vbias_term = (v * self.v_bias).sum(dim=1)  # shape: [batch_size]
        # Compute the activation of the hidden units
        wx_b = F.linear(v, self.W, self.h_bias)     # shape: [batch_size, n_hidden]
        # Compute the hidden term
        hidden_term = torch.sum(torch.log1p(torch.exp(wx_b)), dim=1)  # shape: [batch_size]
        # Return the mean free energy over the batch
        return - (vbias_term + hidden_term).mean()
    
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

def objective(trial):
    num_rbm_epochs = trial.suggest_int("num_rbm_epochs", 24, 33)
    batch_size = trial.suggest_int("batch_size", 192, 1024)
    rbm_lr = trial.suggest_float("rbm_lr", 0.05, 0.1)
    rbm_hidden = trial.suggest_int("rbm_hidden", 384, 8192)

    if CLASSIFIER != 'LogisticRegression':
        fnn_hidden = trial.suggest_int("fnn_hidden", 192, 384)
        fnn_lr = trial.suggest_float("fnn_lr", 0.0001, 0.0025)
        mlflow.log_param("fnn_hidden", fnn_hidden)
        mlflow.log_param("fnn_lr", fnn_lr)

    num_classifier_epochs = trial.suggest_int("num_classifier_epochs", 40, 60)

    mlflow.start_run(experiment_id=experiment.experiment_id)
    mlflow.log_param("num_rbm_epochs", num_rbm_epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("rbm_lr", rbm_lr)
    mlflow.log_param("rbm_hidden", rbm_hidden)
    mlflow.log_param("num_classifier_epochs", num_classifier_epochs)

    # Instantiate RBM and optimizer
    device = torch.device("mps")
    rbm = RBM(n_visible=784, n_hidden=rbm_hidden, k=1).to(device)
    optimizer = torch.optim.SGD(rbm.parameters(), lr=rbm_lr)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    rbm_training_failed = False
    # Training loop (assuming train_loader yields batches of images and labels)
    for epoch in range(num_rbm_epochs):
        total_loss = 0.0
        for images, _ in train_loader:
            # Flatten images and binarize
            v0 = images.view(-1, 784).to(rbm.W.device)      # shape [batch_size, 784]
            v0 = torch.bernoulli(v0)                        # sample binary input
            vk = rbm(v0)                                    # k-step CD reconstruction
            # Compute contrastive divergence loss (free energy difference)
            loss = rbm.free_energy(v0) - rbm.free_energy(vk)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: avg free-energy loss = {total_loss/len(train_loader):.4f}")
        if np.isnan(total_loss):
            rbm_training_failed = True
            break

    if rbm_training_failed:
        accuracy = 0.0
    else:
        rbm.eval()  # set in evaluation mode if using any layers that behave differently in training
        features_list = []
        labels_list = []
        for images, labels in train_loader:
            v = images.view(-1, 784).to(rbm.W.device)
            v = v  # (optionally binarize or use raw normalized pixels)
            h_prob, h_sample = rbm.sample_h(v)  # get hidden activations
            features_list.append(h_prob.cpu().detach().numpy())
            labels_list.append(labels.numpy())
        train_features = np.concatenate(features_list)  # shape: [N_train, n_hidden]
        train_labels = np.concatenate(labels_list)

        # Convert pre-extracted training features and labels to tensors and create a DataLoader
        train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
        train_feature_dataset = torch.utils.data.TensorDataset(train_features_tensor, train_labels_tensor)
        train_feature_loader = torch.utils.data.DataLoader(train_feature_dataset, batch_size=batch_size, shuffle=True)

        if CLASSIFIER == 'LogisticRegression':
            classifier = LogisticRegression(max_iter=num_classifier_epochs)
            classifier.fit(train_features, train_labels)
        else:
            classifier = nn.Sequential(
                nn.Linear(rbm.n_hidden, fnn_hidden),
                nn.ReLU(),
                nn.Linear(fnn_hidden, 10)
            )

            # Move classifier to the same device as the RBM
            classifier = classifier.to(device)
            criterion = nn.CrossEntropyLoss()
            classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=fnn_lr)

            classifier.train()
            for epoch in range(num_classifier_epochs):
                running_loss = 0.0
                for features, labels in train_feature_loader:
                    features = features.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass through classifier
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)
                    
                    # Backpropagation and optimization
                    classifier_optimizer.zero_grad()
                    loss.backward()
                    classifier_optimizer.step()
                    
                    running_loss += loss.item()
                avg_loss = running_loss / len(train_feature_loader)
                print(f"Classifier Epoch {epoch+1}: loss = {avg_loss:.4f}")

        # Evaluate the classifier on test data.
        # Here we extract features from the RBM for each test image.
        if CLASSIFIER != 'LogisticRegression':
            classifier.eval()
        features_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels in test_loader:
                v = images.view(-1, 784).to(device)
                # Extract hidden activations; you can use either h_prob or h_sample.
                h_prob, _ = rbm.sample_h(v)
                features_list.append(h_prob.cpu().detach().numpy())
                labels_list.append(labels.numpy())
        test_features = np.concatenate(features_list)
        test_labels = np.concatenate(labels_list)

        if CLASSIFIER == 'LogisticRegression':
            predictions = classifier.predict(test_features)
            accuracy = accuracy_score(test_labels, predictions) * 100
        else:
            correct = 0
            total = 0
            for features, labels in zip(test_features, test_labels):
                features = torch.tensor(features, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device)
                outputs = classifier(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels).sum().item()
            accuracy = 100 * correct / total

        print(f"Test Accuracy: {accuracy:.2f}%")

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.end_run()

    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)