import time
from PIL import Image
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

train_image_path = 'CSL/train/'
test_image_path = 'CSL/test/'

train_label_path = 'train.txt'
test_label_path = 'test.txt'

size = 64

def get_name_label(file_path):
    name_list = []
    label_list = []
    with open(file_path) as f:                               # f -> "a001.jpg 0"
        for line in f.readlines():
            name_list.append(line.split(' ')[0])             # a001.jpg
            label_list.append(line.split(' ')[1])            # 0
    return name_list,label_list

def get_image_list(filePath,nameList):
    img_list = []
    for name in nameList:
        temp = Image.open(filePath + name)
        temp = temp.resize((size, size), Image.BILINEAR)     # resize image size to 64*64
        img_list.append(temp.copy())
        temp.close()
    return img_list


# get feature by HOG
def get_feature(image_list, name_list, label_list, size):
    i = 0
    fds = []
    labels = []

    for image in image_list:
        try:
            image = np.reshape(image, (size, size, 3))
        except:
            print (name_list[i])
            continue
        gray = rgb_to_gray(image)/255.0                      # normalize
        fd = hog(gray, orientations=9, pixels_per_cell=[16,16], cells_per_block=[2,2], transform_sqrt=True, block_norm='L2-Hys')

        fds.append(fd)
        labels.append(int(label_list[i]))
        i += 1

    return fds, labels

def rgb_to_gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

# train by svm and predict
def train_and_test(fd_train, label_train, fd_test, label_test, SVC_C, SVC_kernel, SVC_degree):
    t0 = time.time()
    num = 0
    total = 0
    
    print ("Training a SVM Classifier.")
    clf = SVC(C=SVC_C, kernel=SVC_kernel, degree=SVC_degree)
    clf.fit(fd_train, label_train)    # fd_train -> 2d array , label_train -> 1d array

    print ("Predicting...")
    result = clf.predict(fd_test)
    print(classification_report(label_test, result))
    
    for test in label_test:
        result = clf.predict(fd_test[total].reshape((1,-1)))
        total += 1
        if int(result[0] == test):
            num += 1

    rate = float(num)/total
    t1 = time.time()
    
    print ('Accuracy : ' + str(rate) + ' (' + str(num) + '/' + str(total) + ')')
    print ('time     : %f sec'%(t1-t0))
    

if __name__ == '__main__':

    t0 = time.time()

    train_name, train_label = get_name_label(train_label_path)
    test_name, test_label = get_name_label(test_label_path)

    train_image = get_image_list(train_image_path,train_name)
    test_image = get_image_list(test_image_path,test_name)

    print('Starting extract feature...')
    fd_train, label_train = get_feature(train_image, train_name, train_label, size)
    fd_train = np.asarray(fd_train)
    label_train = np.asarray(label_train)
    print ("Train features are extracted.")
    fd_test, label_test = get_feature(test_image, test_name, test_label, size)
    fd_test = np.asarray(fd_test)
    label_test = np.asarray(label_test)
    print ("Test features are extracted.")

    t1 = time.time()
    print('time :%f sec'%(t1-t0))


    print('----------Cross Validation:-----------')
    skf = StratifiedKFold(n_splits=5)                # 5-fold cross validation

    R = 0
    P = 0
    fold_cou = 0

    SVC_C = 1.0
    SVC_kernel = 'rbf'
    SVC_degree = 3

    for train, test in skf.split(fd_train, label_train):
        fold_cou += 1
        X_train, X_test = fd_train[train], fd_train[test]
        y_train, y_test = label_train[train], label_train[test]
        clf = SVC(C=SVC_C, kernel=SVC_kernel, degree=SVC_degree)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        r = precision_score(y_test, y_pred, average='weighted')
        p = recall_score(y_test, y_pred, average='weighted')
        #print(classification_report(y_test, y_pred))
        print('fold : %d\trecall : %f\tprecision : %f'%(fold_cou, r, p))
        R += r
        P += p
    print('C = %f\tkernel = %s\tdegree = %d' %(SVC_C, SVC_kernel, SVC_degree))
    print('ave_recall   : %f' %(R/5.0))
    print('ave_precision: %f' %(P/5.0))
        

    train_and_test(fd_train, label_train, fd_test, label_test, SVC_C, SVC_kernel, SVC_degree)
