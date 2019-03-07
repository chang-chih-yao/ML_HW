import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import MNIST_tools
import timeit

def Relu(x):
    return x * (x > 0)

def dRelu(x):
    return 1. * (x > 0)

def Softmax(x):
    e_x = np.exp(x.T - np.max(x, axis = -1))
    return (e_x / e_x.sum(axis=0)).T

def Sigmoid(x):
    return 1. / (1. + np.exp(-x))


def No_1(epoch=100, batch_size=1000, lr=0.00001):
    ''' Initial variable '''
    W1 = np.random.randn(784, 256)/784
    b1 = np.random.randn(256)
    W2 = np.random.randn(256, 10)/256
    b2 = np.random.randn(10)

    arr_E = np.array([])
    arr_A_train = np.array([])
    arr_A_test = np.array([])

    current_E = 9999
    current_Acc_train = 0
    current_Acc_test = 0

    for i in range(epoch):
        now_batch = 0
        while(now_batch < x_train.shape[0]):
            if (now_batch + batch_size) > x_train.shape[0]:
                x_batch = x_train[now_batch:60000]
            else:
                x_batch = x_train[now_batch:now_batch + batch_size]

            ''' Feedforward '''
            A1 = b1 + np.dot(x_batch, W1)            # (batch_size, 256)
            H = Relu(A1)
            A2 = b2 + np.dot(H, W2)                  # (batch_size, 10)
            Y = Softmax(A2)

            ''' calculate accuracy '''
            compare_arr = np.equal(np.argmax(Y, axis=1), np.argmax(y_train[now_batch:now_batch + batch_size], axis=1))
            count = 0
            for item in compare_arr:
                if item == True:
                    count += 1
            current_Acc_train = float(count/batch_size*100)

            result = Softmax(b2 + np.dot(Relu(b1 + np.dot(x_test, W1)), W2))
            compare_arr = np.equal(np.argmax(result, axis=1), np.argmax(y_test, axis=1))
            count = 0
            for item in compare_arr:
                if item == True:
                    count += 1
            current_Acc_test = float(count/x_test.shape[0]*100)

            E = -np.sum(np.log(Y + 1e-19) * y_train[now_batch : now_batch + batch_size], axis=1)   # (batch_size, )
            current_E = E

            ''' Backprop '''
            dSoftmax = Y - y_train[now_batch : now_batch + batch_size]   # (batch_size, 10)
            dW2 = np.dot(H.T, dSoftmax)                                  # (256, 10)
            db2 = np.dot(np.ones((1, batch_size)), dSoftmax)             # (1, 10)

            dH = np.dot(dSoftmax, W2.T)                                  # (batch_size, 256)
            dA = dH * dRelu(A1)                                          # (batch_size, 256)

            dW1 = np.dot(x_batch.T, dA)                                  # (784, 256)
            db1 = np.dot(np.ones((1, batch_size)), dA)                   # (1, 256)
            
            ''' update W and b '''
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2.flatten()
            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1.flatten()

            now_batch += batch_size

        print('%d/%d : %f, Accuracy_train : %f, Accuracy_test : %f' % (i + 1, epoch, np.mean(current_E), current_Acc_train, current_Acc_test))
        arr_E = np.append(arr_E, np.mean(current_E))
        arr_A_train = np.append(arr_A_train, current_Acc_train)
        arr_A_test = np.append(arr_A_test, current_Acc_test)
    plt.title('No_1_Error')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.plot(np.arange(epoch), arr_E, label='Error')
    plt.legend()
    plt.show()

    plt.title('No_1_Accuracy')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy(%)', fontsize=14)
    plt.plot(np.arange(epoch), arr_A_train, label='Accuracy_train')
    plt.plot(np.arange(epoch), arr_A_test, label='Accuracy_test')
    plt.legend()
    plt.show()

def No_2(epoch=200, batch_size=1000, lr=0.00002):
    ''' Initial variable '''
    W1 = np.random.randn(784, 204)/784
    b1 = np.random.randn(204)
    W2 = np.random.randn(204, 202)/204
    b2 = np.random.randn(202)
    W3 = np.random.randn(202, 10)/202
    b3 = np.random.randn(10)

    arr_E = np.array([])
    arr_A_train = np.array([])
    arr_A_test = np.array([])

    current_E = 9999
    current_Acc_train = 0
    current_Acc_test = 0

    for i in range(epoch):
        now_batch = 0
        while(now_batch < x_train.shape[0]):
            if (now_batch + batch_size) > x_train.shape[0]:
                x_batch = x_train[now_batch:60000]
            else:
                x_batch = x_train[now_batch:now_batch + batch_size]

            ''' Feedforward '''
            A1 = b1 + np.dot(x_batch, W1)             # (batch_size, 204)
            H1 = Relu(A1)
            A2 = b2 + np.dot(H1, W2)                  # (batch_size, 202)
            H2 = Relu(A2)
            A3 = b3 + np.dot(H2, W3)                  # (batch_size, 10)
            Y = Softmax(A3)

            ''' calculate accuracy '''
            compare_arr = np.equal(np.argmax(Y, axis=1), np.argmax(y_train[now_batch:now_batch + batch_size], axis=1))
            count = 0
            for item in compare_arr:
                if item == True:
                    count += 1
            current_Acc_train = float(count/batch_size*100)

            result = Softmax(b3 + np.dot(Relu(b2 + np.dot(Relu(b1 + np.dot(x_test, W1)), W2)), W3))
            compare_arr = np.equal(np.argmax(result, axis=1), np.argmax(y_test, axis=1))
            count = 0
            for item in compare_arr:
                if item == True:
                    count += 1
            current_Acc_test = float(count/x_test.shape[0]*100)

            E = -np.sum(np.log(Y + 1e-19) * y_train[now_batch : now_batch + batch_size], axis=1)   # (batch_size, )
            #print(np.mean(E))
            current_E = E

            ''' Backprop '''
            dSoftmax = Y - y_train[now_batch : now_batch + batch_size]   # (batch_size, 10)
            dW3 = np.dot(H2.T, dSoftmax)                                 # (202, 10)
            db3 = np.dot(np.ones((1, batch_size)), dSoftmax)             # (1, 10)

            dH2 = np.dot(dSoftmax, W3.T)                                 # (batch_size, 202)
            dA2 = dH2 * dRelu(A2)                                        # (batch_size, 202)
            dW2 = np.dot(H1.T, dA2)                                      # (204, 202)
            db2 = np.dot(np.ones((1, batch_size)), dA2)                  # (1, 202)

            dH1 = np.dot(dA2, W2.T)                                      # (batch_size, 204)
            dA1 = dH1 * dRelu(A1)                                        # (batch_size, 204)
            dW1 = np.dot(x_batch.T, dA1)                                 # (784, 204)
            db1 = np.dot(np.ones((1, batch_size)), dA1)                  # (1, 204)
            
            ''' update W and b '''
            W3 = W3 - lr * dW3
            b3 = b3 - lr * db3.flatten()
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2.flatten()
            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1.flatten()

            now_batch += batch_size

        print('%d/%d : %f, Accuracy_train : %f, Accuracy_test : %f' % (i + 1, epoch, np.mean(current_E), current_Acc_train, current_Acc_test))
        arr_E = np.append(arr_E, np.mean(current_E))
        arr_A_train = np.append(arr_A_train, current_Acc_train)
        arr_A_test = np.append(arr_A_test, current_Acc_test)
    plt.title('No_2_Error')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.plot(np.arange(epoch), arr_E, label='Error')
    plt.legend()
    plt.show()

    plt.title('No_2_Accuracy')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy(%)', fontsize=14)
    plt.plot(np.arange(epoch), arr_A_train, label='Accuracy_train')
    plt.plot(np.arange(epoch), arr_A_test, label='Accuracy_test')
    plt.legend()
    plt.show()

def No_3(epoch=100, batch_size=1000, lr=0.00001):
    ''' Initial variable '''
    W1 = np.random.randn(784, 128)/784
    b1 = np.random.randn(128)
    W2 = np.random.randn(128, 784)/128
    b2 = np.random.randn(784)

    current_E = 9999

    for i in range(epoch):
        now_batch = 0
        while(now_batch < x_train.shape[0]):
            if (now_batch + batch_size) > x_train.shape[0]:
                x_batch = x_train[now_batch:60000]
            else:
                x_batch = x_train[now_batch:now_batch + batch_size]

            ''' Feedforward '''
            A1 = b1 + np.dot(x_batch, W1)            # (batch_size, 128)
            H = Relu(A1)
            A2 = b2 + np.dot(H, W2)                  # (batch_size, 784)
            Y = Sigmoid(A2)

            E = -np.sum(np.log(Y + 1e-19) * x_train[now_batch : now_batch + batch_size], axis=1)   # (batch_size, )
            #print(np.mean(E))
            current_E = E

            ''' Backprop '''
            dSigmoid = Y - x_train[now_batch : now_batch + batch_size]   # (batch_size, 784)
            dW2 = np.dot(H.T, dSigmoid)                                  # (128, 784)
            db2 = np.dot(np.ones((1, batch_size)), dSigmoid)             # (1, 784)

            dH = np.dot(dSigmoid, W2.T)                                  # (batch_size, 128)
            dA1 = dH * dRelu(A1)                                          # (batch_size, 128)

            dW1 = np.dot(x_batch.T, dA1)                                  # (784, 128)
            db1 = np.dot(np.ones((1, batch_size)), dA1)                   # (1, 128)
            
            ''' update W and b '''
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2.flatten()
            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1.flatten()

            now_batch += batch_size

        print('%d/%d : %f' % (i + 1, epoch, np.mean(current_E)))
    
    dim_red = b1 + np.dot(x_test, W1)
    x_ = dim_red[:, 0]
    y_ = dim_red[:, 1]
    for i in range(10):
        x_dim = x_[y_test_digit==i]
        y_dim = y_[y_test_digit==i]
        x = x_test[y_test_digit==i]
        draw_image = np.reshape(x, (len(x), 28, 28))
        now_image = np.array([], dtype='int32')
        fig, ax = plt.subplots(figsize=(10, 9))
        plt.title('Dimension reduction')
        ax.plot(x_dim, y_dim, 'b.', markersize=1)

        xy = (x_dim[0], y_dim[0])
        imagebox = OffsetImage(draw_image[0], zoom=0.4, cmap=plt.cm.binary)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, xy, xybox=(10, 10), xycoords='data', boxcoords="offset points")
        ax.add_artist(ab)

        ax.plot(x_dim[0], y_dim[0], 'ro', markersize=2)

        now_image = np.append(now_image, 0)

        flag = True

        for j in range(1, len(x)):
            flag = True
            for item in now_image:
                if ((x_dim[item] - x_dim[j])**2 + (y_dim[item] - y_dim[j])**2) < 0.05:
                    flag = False
                    break
            if(flag):    # if flag is True, means this point is not close to other point
                xy = (x_dim[j], y_dim[j])
                imagebox = OffsetImage(draw_image[j], zoom=0.4, cmap=plt.cm.binary)
                imagebox.image.axes = ax
                ab = AnnotationBbox(imagebox, xy, xybox=(10, 10), xycoords='data', boxcoords="offset points")
                ax.add_artist(ab)

                ax.plot(x_dim[j], y_dim[j], 'ro', markersize=2)

                now_image = np.append(now_image, j)   # now_image : which image(indices) are drawing on the board

        plt.show()

    result = Sigmoid(b2 + np.dot(Relu(b1 + np.dot(x_test, W1)), W2))
    plt.title('Reconstruction')
    plt.subplot(2, 3, 1)
    plt.imshow(x_test[0].reshape([28,28]), cmap="gray")
    plt.subplot(2, 3, 2)
    plt.imshow(x_test[1].reshape([28,28]), cmap="gray")
    plt.subplot(2, 3, 3)
    plt.imshow(x_test[2].reshape([28,28]), cmap="gray")
    plt.subplot(2, 3, 4)
    plt.imshow(result[0].reshape([28,28]), cmap="gray")
    plt.subplot(2, 3, 5)
    plt.imshow(result[1].reshape([28,28]), cmap="gray")
    plt.subplot(2, 3, 6)
    plt.imshow(result[2].reshape([28,28]), cmap="gray")
    plt.show()

    plt.title('Filter')
    for j in range(9):
        plt.subplot(3, 3, j+1)
        plt.imshow(W1[:, j].reshape([28, 28]), cmap="gray")
    plt.show()


if __name__ == "__main__":
    MNIST_tools.downloadMNIST(path='MNIST_data', unzip=True)
    x_train, y_train = MNIST_tools.loadMNIST(dataset="training", path="MNIST_data")
    x_test, y_test = MNIST_tools.loadMNIST(dataset="testing", path="MNIST_data")
    
    print(x_train.shape)                    # (60000, 784)
    y_train_digit = y_train
    y_test_digit = y_test

    ''' normalize '''
    x_train = x_train.astype('float')
    x_train = x_train / 255.0
    x_test = x_test.astype('float')
    x_test = x_test / 255.0

    ''' change to one_hot '''
    b = np.zeros((y_train.shape[0], 10))
    b[np.arange(y_train.shape[0]), y_train] = 1
    y_train = b
    b = np.zeros((y_test.shape[0], 10))
    b[np.arange(y_test.shape[0]), y_test] = 1
    y_test = b

    No_1(epoch=100, batch_size=6000, lr=0.00001)
    No_2(epoch=100, batch_size=1000, lr=0.00003)
    No_3(epoch=70, batch_size=600, lr=0.00001)
