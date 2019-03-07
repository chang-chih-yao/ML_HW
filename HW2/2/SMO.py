import numpy as np
from sklearn.metrics import accuracy_score

class SVM():
    def __init__(self, C=1.0, kernel='linear', max_iter=1000, epsilon=0.001):
        self.kernels = {
            'linear' : self.kernel_linear,
            'rbf' : self.kernel_rbf
        }
        self.max_iter = max_iter
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon

    def fit(self, X, y):          # X.shape -> (144, )
        n = X.shape[0]            # n -> number of dataset
        alpha = np.zeros((n))
        kernel = self.kernels[self.kernel]
        print(kernel)
        count = 0

        while True:
            count += 1
            alpha_prev = np.copy(alpha)
            
            for j in range(0, n-1):

                i = j
                while i==j:
                    i = np.random.randint(0, n-1)            # random pick i

                x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]

                # Compute L & H, the bounds on new possible alpha values
                if(y_i != y_j):
                    L = max(0, alpha_prime_j - alpha_prime_i)
                    H = min(self.C, self.C - alpha_prime_i + alpha_prime_j)
                else:
                    L = max(0, alpha_prime_i + alpha_prime_j - self.C)
                    H = min(self.C, alpha_prime_i + alpha_prime_j)


                eta = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)
                
                if eta == 0:
                    continue

                # Compute model parameters
                self.w = np.dot(X.T, np.multiply(alpha, y))
                self.b = np.mean(y - np.dot(self.w.T, X.T))

                # Compute E_i, E_j
                # Prediction error
                E_i = self.h(x_i, self.w, self.b) - y_i
                E_j = self.h(x_j, self.w, self.b) - y_j

                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j)) / eta
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

            # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

            if count >= self.max_iter:
                print("max of %d iterations" % (self.max_iter))
                return

        np.set_printoptions(precision=6, suppress=True)  #print without scientific notation
        print('alpha:')
        print(alpha)

        # Compute final model parameters
        self.b = np.mean(y - np.dot(self.w.T, X.T))
        if self.kernel == 'linear':
            self.w = np.dot(X.T, np.multiply(alpha, y))
        
        return count

    # Prediction
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    def predict(self, X):
        return self.h(X, self.w, self.b)

    # Define kernels
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T) + 1

    def kernel_rbf(self, x1, x2, sigma=1):
        result = np.exp(- np.linalg.norm(x1 - x2) / (2 * sigma ** 2))
        return result


def read_file():
    with open('liver-disorders_scale.txt', 'r') as f:
        data = np.array([])
        for line in f.readlines():
            if len(line.split(' ')) == 7:              # only one data miss one feature(<7)
                if line.split(' ')[0] == '0':
                    data = np.append(data, -1)
                else:
                    data = np.append(data, 1)
                for i in range(1, 6):
                    data = np.append(data, float(line.split(' ')[i].split(':')[1]))
        data = np.reshape(data, (144, 6))
    return data


if __name__ == '__main__':

    data = read_file()
    X, y = data[:,1:], data[:,0].astype(int)       # X.shape -> (144, 5)      y.shape -> 144

    model = SVM(C=10.0, kernel='rbf')              # default C = 1.0 , kernel = 'linear'
    iterations = model.fit(X, y)

    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)

    print("accuracy : %f" % (acc))
    #print("Converged after %d iterations" % (iterations))