import numpy as np


class LinearRegression:
    def __init__(
        self,
        *,
        penalty="l2",
        alpha=0.0001,
        max_iter=1000,
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
        self.loss_history = []
        self.validation_loss_history = []

    def get_penalty_grad(self):

        if self.penalty == "l2":
            return 2*self.alpha * self.coef_
        elif self.penalty == "l1":
            return self.alpha * np.sign(self.coef_)
        else:
            return 0

    def fit(self, x, y):
        np.random.seed(self.random_state)
        n_samples, n_features = x.shape
        self.coef_ = np.random.randn(n_features)
        self.intercept_ = np.random.randn()

        if self.early_stopping:
            n_validation = int(n_samples * (1 - self.validation_fraction))
            x_train, y_train = x[:-n_validation], y[:-n_validation]
            x_val, y_val = x[-n_validation:], y[-n_validation:]
        else:
            x_train, y_train = x, y
            x_val, y_val = None, None

        best_loss = np.inf
        no_improvement_count = 0
        for epoch in range(self.max_iter):
            if self.shuffle:
                p = np.random.permutation(len(x_train))
                x_train, y_train = x_train[p], y_train[p]

            for i in range(0, len(x_train), self.batch_size):
                x_batch = x_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                y_pred = x_batch @ self.coef_ + self.intercept_

                error = y_pred - y_batch
                grad_coef = (2 / len(y_batch))*(x_batch.T @
                                                error) + self.get_penalty_grad()
                grad_intercept = (2 / len(y_batch)) * np.sum(error)
                self.coef_ -= self.eta0 * grad_coef
                self.intercept_ -= self.eta0 * grad_intercept

            train_loss = self._compute_loss(x_train, y_train)
            self.loss_history.append(train_loss)

            if self.early_stopping:
                y_val_pred = x_val @ self.coef_ + self.intercept_
                val_loss = np.mean((y_val_pred - y_val) ** 2)

                if val_loss < best_loss - self.tol:
                    best_loss = val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= self.n_iter_no_change:
                    break

    def predict(self, x):
        return x @ self.coef_ + self.intercept_

    def _compute_loss(self, x, y):
        y_pred = self.predict(x)
        mse = np.mean((y_pred - y) ** 2)
        if self.penalty == "l2":
            mse += 2*self.alpha * np.sum(self.coef_ ** 2)
        elif self.penalty == "l1":
            mse += self.alpha * np.sum(np.abs(self.coef_))
        return mse

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
