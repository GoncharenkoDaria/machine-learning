import numpy as np


class SoftmaxRegression:
    def __init__(
        self,
        *,
        penalty="l2",
        alpha=0.0001,
        max_iter=100,
        tol=0.001,
        random_state=None,
        eta0=0.01,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        shuffle=True,
        batch_size=32
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eta0 = eta0
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.batch_size = batch_size
        self._coef = None
        self._intercept = None
        self.classes_ = None
        self.n_features_in_ = None
        self._rng = np.random.RandomState(random_state)
        self._best_loss = np.inf
        self._no_improvement_count = 0

    def get_penalty_grad(self):
        if self._coef is None:
            return np.zeros_like(self._coef) if hasattr(self, '_coef') else 0

        if self.penalty == "l2":
            return 2*self.alpha * self._coef
        elif self.penalty == "l1":
            return self.alpha * np.sign(self._coef)
        return np.zeros_like(self._coef)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), np.searchsorted(self.classes_, y)] = 1

        self._coef = self._rng.normal(scale=0.01, size=(n_classes, n_features))
        self._intercept = np.zeros(n_classes)

        if self.early_stopping:
            val_size = int(self.validation_fraction * n_samples)
            X_train, X_val = X[:-val_size], X[-val_size:]
            y_train, y_val = y_onehot[:-val_size], y_onehot[-val_size:]
        else:
            X_train, y_train = X, y_onehot
            X_val, y_val = None, None

        best_weights = (self._coef.copy(), self._intercept.copy())

        for epoch in range(self.max_iter):
            if self.shuffle:
                indices = self._rng.permutation(len(X_train))
                X_train = X_train[indices]
                y_train = y_train[indices]

            epoch_grad_norm = 0.0
            batch_count = 0

            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                logits = X_batch @ self._coef.T + self._intercept
                probs = self.softmax(logits)

                error = probs - y_batch
                grad_coef = 2*error.T @ X_batch / len(X_batch)
                grad_intercept = 2*np.mean(error, axis=0)

                penalty_grad = self.get_penalty_grad()
                grad_coef += penalty_grad

                total_grad = np.concatenate(
                    [grad_coef.ravel(), grad_intercept])
                epoch_grad_norm += np.linalg.norm(total_grad)
                batch_count += 1

                self._coef -= self.eta0 * grad_coef
                self._intercept -= self.eta0 * grad_intercept

            avg_grad_norm = epoch_grad_norm / batch_count if batch_count > 0 else 0
            if avg_grad_norm < self.tol:
                break

            if self.early_stopping and X_val is not None:
                val_probs = self.predict_proba(X_val)
                loss = -np.mean(np.log(
                    val_probs[np.arange(len(X_val)), np.argmax(
                        y_val, axis=1)] + 1e-10
                ))

                if loss < self._best_loss - self.tol:
                    self._best_loss = loss
                    best_weights = (self._coef.copy(), self._intercept.copy())
                    self._no_improvement_count = 0
                else:
                    self._no_improvement_count += 1
                    if self._no_improvement_count >= self.n_iter_no_change:
                        self._coef, self._intercept = best_weights
                        break

        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        logits = X @ self._coef.T + self._intercept
        return self.softmax(logits)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @staticmethod
    def softmax(z):
        e_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e_z / np.sum(e_z, axis=-1, keepdims=True)

    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept

    @coef_.setter
    def coef_(self, value):
        self._coef = value

    @intercept_.setter
    def intercept_(self, value):
        self._intercept = value
