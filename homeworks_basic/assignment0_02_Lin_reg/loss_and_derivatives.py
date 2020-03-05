import numpy as np


class LossAndDerivatives:
    @staticmethod
    def mse(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`)
        w : numpy array of shape (`n_features`, `target_dimentionality`)

        Return : float
            single number with MSE value of linear model (X.dot(w)) with no bias term
            on the selected dataset.
        
        Comment: If Y is two-dimentional, average the error over both dimentions.
        """

        return np.mean((X.dot(w) - Y)**2)

    @staticmethod
    def mae(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`)
        w : numpy array of shape (`n_features`, `target_dimentionality`)
                
        Return: float
            single number with MAE value of linear model (X.dot(w)) with no bias term
            on the selected dataset.

        Comment: If Y is two-dimentional, average the error over both dimentions.
        """
   
        return np.mean(((X.dot(w) - Y)**2)**0.5)

    @staticmethod
    def l2_reg(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`)

        Return: float
            single number with L2 norm of the weight matrix

        Computes the L2 regularization term for the weight matrix w.
        """
        return np.sum(w ** 2)

    @staticmethod
    def l1_reg(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`)

        Return : float
            single number with L1 norm of the weight matrix
        
        Computes the L1 regularization term for the weight matrix w.
        """
        return np.sum((w ** 2) ** 0.5)

    @staticmethod
    def no_reg(w):
        """
        Simply ignores the regularization
        """
        return np.zeros_like(w)
    
    @staticmethod
    def mse_derivative(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`)
        w : numpy array of shape (`n_features`, `target_dimentionality`)
        
        Return : numpy array of shape (`n_features`, `target_dimentionality`)

        Computes the MSE derivative for linear regression (X.dot(w)) with no bias term
        w.r.t. w weight matrix.
        
        Please mention, that in case `target_dimentionality` > 1 the error is averaged along this
        dimension as well, so you need to consider that fact in derivative implementation.
        """

        return 2 * np.dot(X.T, np.dot(X ,w) - Y) / Y.shape[0] / Y.shape[1]

    @staticmethod
    def mae_derivative(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`)
        w : numpy array of shape (`n_features`, `target_dimentionality`)
        
        Return : numpy array of shape (`n_features`, `target_dimentionality`)

        Computes the MAE derivative for linear regression (X.dot(w)) with no bias term
        w.r.t. w weight matrix.
        
        Please mention, that in case `target_dimentionality` > 1 the error is averaged along this
        dimension as well, so you need to consider that fact in derivative implementation.
        """
        Y_prep = np.dot(X, w)
        d = np.ones(Y.shape)
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                d[i,j] = -1 if (Y[i,j] - Y_prep[i,j]) >= 0 else 1
        return  np.dot(X.T, d) / Y.shape[0] / Y.shape[1]

    @staticmethod
    def l2_reg_derivative(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`)

        Return : numpy array of shape (`n_features`, `target_dimentionality`)

        Computes the L2 regularization term derivative w.r.t. the weight matrix w.
        """
        return 2 * w

    @staticmethod
    def l1_reg_derivative(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`)

        Return : numpy array of shape (`n_features`, `target_dimentionality`)

        Computes the L1 regularization term derivative w.r.t. the weight matrix w.
        """
        d = np.ones(w.shape)
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                d[i,j] = 1 if w[i,j] >= 0 else -1
        return d

    @staticmethod
    def no_reg_derivative(w):
        """
        Simply ignores the derivative
        """
        return np.zeros_like(w)
