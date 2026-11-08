"""A small PyTorch feed-forward network wrapped as a scikit-learn classifier, so it can sit inside the
same preprocessing Pipeline, cross-validation and calibration as the tree models."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden=(64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-4, epochs=60,
                 batch_size=128, patience=8, seed=42):
        self.hidden, self.dropout, self.lr, self.weight_decay = hidden, dropout, lr, weight_decay
        self.epochs, self.batch_size, self.patience, self.seed = epochs, batch_size, patience, seed

    def _net(self, d):
        layers, prev = [], d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        return nn.Sequential(*layers, nn.Linear(prev, 1))

    @staticmethod
    def _dense(X):
        return np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X, y = self._dense(X), np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        idx = rng.permutation(len(X)); cut = int(0.85 * len(X))
        tr, va = idx[:cut], idx[cut:]
        Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
        Xv, yv = torch.from_numpy(X[va]), torch.from_numpy(y[va])
        self.model_ = self._net(X.shape[1])
        pos_w = torch.tensor((1 - y.mean()) / y.mean())
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):  # mini-batch training with early stopping on validation loss
            self.model_.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), self.batch_size):
                b = perm[i:i + self.batch_size]
                if len(b) < 2: continue
                opt.zero_grad(); loss_fn(self.model_(Xt[b]).squeeze(1), yt[b]).backward(); opt.step()
            self.model_.eval()
            with torch.no_grad():
                vl = loss_fn(self.model_(Xv).squeeze(1), yv).item()
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience: break
        self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model_(torch.from_numpy(self._dense(X))).squeeze(1)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
