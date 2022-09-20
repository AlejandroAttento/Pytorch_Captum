import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class DenseNN(nn.Module):
    def __init__(self, in_size, hid1_size, hid2_size, out_size, criterion, optimizer, dropout=0.1, learning_rate=0.02):
        super().__init__()
        self.z1 = nn.Linear(in_size, hid1_size)
        self.a1 = nn.ReLU()
        self.d1 = nn.Dropout(p=dropout)
        self.z2 = nn.Linear(hid1_size, hid2_size)
        self.a2 = nn.ReLU()
        self.d2 = nn.Dropout(p=dropout)
        self.z3 = nn.Linear(hid2_size, out_size)
        self.output = nn.Sigmoid()

        self.criterion = criterion
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f'{optimizer} is not an valid optimizer.')

    def forward(self, x):
        z1 = self.z1(x)
        a1 = self.a1(z1)
        d1 = self.d1(a1)

        z2 = self.z2(d1)
        a2 = self.a2(z2)
        d2 = self.d2(a2)

        z3 = self.z3(d2)

        return self.output(z3)

    def training_step(self, data):
        features, labels = data

        self.optimizer.zero_grad()

        preds = self(features)

        loss = self.criterion(torch.flatten(preds), labels)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def testing_step(self, data):
        features, labels = data

        preds = self(features)
        loss = self.criterion(torch.flatten(preds), labels)

        return loss.item()

    def _model_metrics(self, preds, labels):
        preds = (preds.detach().numpy() > 0.5).astype('int')
        labels = (np.vectorize(lambda x: False if x == 0 else True)(labels.numpy())).astype('int')

        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        accuracy = accuracy_score(labels, preds)

        return {'f1':f1, 'precision':precision, 'recall':recall, 'accuracy':accuracy}

    def model_eval(self, train_features, train_labels, test_features, test_labels):
        train_preds = self(train_features)
        test_preds = self(test_features)

        train_metrics = self._model_metrics(train_preds, train_labels)
        test_metrics = self._model_metrics(test_preds, test_labels)

        return {'train_metrics':train_metrics, 'test_metrics':test_metrics}
