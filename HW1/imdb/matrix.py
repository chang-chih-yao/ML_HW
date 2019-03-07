import numpy as np
import math

X_train = np.load("x_train.npy")
Y_train = np.load("y_train.npy")
X_test = np.load("x_test.npy")
Y_test = np.load("y_test.npy")

K = 10000
negative_num = 0
word = {}
word[0] = [0, 0, 0]

for i in range(len(X_train)):
    temp_arr = {}
    if Y_train[i] == 0:
        negative_num += 1
    for j in range(len(X_train[i])):   
        if X_train[i][j] <= K:
            if Y_train[i]==0:
                temp_arr[X_train[i][j]] = 0
            else:
                temp_arr[X_train[i][j]] = 1
    for v in temp_arr:
        if temp_arr[v] == 0:
            if v in word:
                word[v][0] += 1
            else:
                word[v] = [0, 0]
                word[v][0] += 1
        else:
            if v in word:
                word[v][1] += 1
            else:
                word[v] = [0, 0]
                word[v][1] += 1

positive_num = len(X_train) - negative_num
#print(word)
#print(negative_num)
#print(word[1])

f = 0
t = 0
correct = 0
TP = 0
FP = 0
FN = 0
TN = 0

def predict_T_or_F(f, t):
    if f>t:
        return 0
    else:
        return 1

for i in range(len(X_test)):
    f = 0
    t = 0
    temp_arr2 = {}
    for j in range(len(X_test[i])):
        if X_test[i][j] <= K:
            temp_arr2[X_test[i][j]] = 1

    for v in temp_arr2:
        f += math.log(1e-9 + word[v][0] / negative_num, 2)
        t += math.log(1e-9 + word[v][1] / positive_num, 2)
    f += math.log(1e-9 + negative_num/len(X_test), 2)
    t += math.log(1e-9 + positive_num/len(X_test), 2)

    if predict_T_or_F(f, t) == Y_test[i]:
        correct += 1
    if (predict_T_or_F(f, t) == 1 and predict_T_or_F(f, t) == Y_test[i]):
        TP += 1
    if (predict_T_or_F(f, t) == 1 and predict_T_or_F(f, t) != Y_test[i]):
        FP += 1
    if (predict_T_or_F(f, t) == 0 and predict_T_or_F(f, t) != Y_test[i]):
        FN += 1
    if (predict_T_or_F(f, t) == 0 and predict_T_or_F(f, t) == Y_test[i]):
        TN += 1

print("K =", K)
print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("TN:", TN)
print("Accuracy:", correct, "/",  len(X_test), "=", correct/len(X_test))
print("Precision:", TP/(TP+FP))
print("Recall:", TP/(TP+FN))