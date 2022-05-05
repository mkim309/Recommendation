import numpy as np

def recommender(rate_mat, lr, with_reg):
    """

    :param rate_mat:
    :param lr:
    :param with_reg:
        boolean flag, set true for using regularization and false otherwise
    :return:
    """

    # TODO pick hyperparams
    max_iter = 200
    learning_rate = 0.0004
    reg_coef = 0
    n_user, n_item = rate_mat.shape[0], rate_mat.shape[1]

    U = np.random.rand(n_user, lr) / lr
    V = np.random.rand(n_item, lr) / lr

    # TODO implement your code here
    if with_reg:
        reg_coef = 0.6
        for i in range(max_iter):
            A = (rate_mat > 0) * (rate_mat - np.matmul(U, V.T))
            U_temp = U + 2 * learning_rate * np.matmul(A, V) - 2 * learning_rate * U
            V_temp = V + 2 * learning_rate * np.matmul(A.T, U) - 2 * learning_rate * V
            U = np.copy(U_temp)
            V = np.copy(V_temp)

    else:
        for i in range(max_iter):
            A = (rate_mat > 0) * (rate_mat - np.matmul(U, V.T))
            U_temp = U + 2 * learning_rate * np.matmul(A, V) 
            V_temp = V + 2 * learning_rate * np.matmul(A.T, U)
            U = np.copy(U_temp)
            V = np.copy(V_temp)


    return U, V
