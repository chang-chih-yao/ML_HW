from tensorflow.examples.tutorials.mnist import input_data
import sklearn.decomposition as tp
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import numpy as np

data = input_data.read_data_sets('MNIST_data', one_hot=True)
x_data = data.train.images
y_data = data.train.labels
print(x_data.shape)
print(y_data.shape)

# for PCA
for i in range(10):
    temp = np.argmax(y_data, axis=1)
    x = x_data[temp == i]
    y = y_data[temp == i]

    pca = tp.PCA(n_components=2)
    data_pca = pca.fit_transform(x)
    pca_x = data_pca[:, 0]
    pca_y = data_pca[:, 1]

    draw_image = np.reshape(x, (len(x), 28, 28))
    now_image = np.array([], dtype='int32')
    
    fig, ax = plt.subplots(figsize=(10, 9))
    s = 'PCA : ' + str(i)
    plt.title(s)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.axis([-7,9,-7,8])

    ax.plot(pca_x, pca_y, 'b.', markersize=1)


    xy = (pca_x[0], pca_y[0])
    imagebox = OffsetImage(draw_image[0], zoom=0.4, cmap=plt.cm.binary)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy, xybox=(10, 10), xycoords='data', boxcoords="offset points")
    ax.add_artist(ab)

    ax.plot(pca_x[0], pca_y[0], 'ro', markersize=2)

    now_image = np.append(now_image, 0)


    flag = True

    for j in range(1, len(x)):
        flag = True
        for item in now_image:
            if ((pca_x[item] - pca_x[j])**2 + (pca_y[item] - pca_y[j])**2) < 1:
                flag = False
                break
        if(flag):    # if flag is True, means this point is not close to other point
            xy = (pca_x[j], pca_y[j])
            imagebox = OffsetImage(draw_image[j], zoom=0.4, cmap=plt.cm.binary)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, xy, xybox=(10, 10), xycoords='data', boxcoords="offset points")
            ax.add_artist(ab)

            ax.plot(pca_x[j], pca_y[j], 'ro', markersize=2)

            now_image = np.append(now_image, j)   # now_image : which image(indices) are drawing on the board

    plt.show()

# for ICA
for i in range(10):
    temp = np.argmax(y_data, axis=1)
    x = x_data[temp == i]
    y = y_data[temp == i]

    ica = tp.FastICA(n_components=2)
    data_ica = ica.fit_transform(x)
    ica_x = data_ica[:, 0]
    ica_y = data_ica[:, 1]

    draw_image = np.reshape(x, (len(x), 28, 28))
    now_image = np.array([], dtype='int32')
    
    fig, ax = plt.subplots(figsize=(10, 9))
    s = 'ICA : ' + str(i)

    plt.title(s)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    #plt.axis([-7,9,-7,8])

    ax.plot(ica_x, ica_y, 'b.', markersize=1)


    xy = (ica_x[0], ica_y[0])
    imagebox = OffsetImage(draw_image[0], zoom=0.4, cmap=plt.cm.binary)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy, xybox=(10, 10), xycoords='data', boxcoords="offset points")
    ax.add_artist(ab)

    ax.plot(ica_x[0], ica_y[0], 'ro', markersize=2)

    now_image = np.append(now_image, 0)


    flag = True

    for j in range(1, len(x)):
        flag = True
        for item in now_image:
            if ((ica_x[item] - ica_x[j])**2 + (ica_y[item] - ica_y[j])**2) < 2e-5:
                flag = False
                break
        if(flag):
            #print(j, item)
            xy = (ica_x[j], ica_y[j])
            imagebox = OffsetImage(draw_image[j], zoom=0.4, cmap=plt.cm.binary)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, xy, xybox=(10, 10), xycoords='data', boxcoords="offset points")
            ax.add_artist(ab)

            ax.plot(ica_x[j], ica_y[j], 'ro', markersize=2)

            now_image = np.append(now_image, j)

    plt.show()

# for LLE
for i in range(10):
    temp = np.argmax(y_data, axis=1)
    x = x_data[temp == i]
    y = y_data[temp == i]

    lle = LocallyLinearEmbedding(n_components=2)
    data_lle = lle.fit_transform(x)
    lle_x = data_lle[:, 0]
    lle_y = data_lle[:, 1]

    draw_image = np.reshape(x, (len(x), 28, 28))
    now_image = np.array([], dtype='int32')
    
    fig, ax = plt.subplots(figsize=(10, 9))
    s = 'LLE : ' + str(i)

    plt.title(s)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    #plt.axis([-7,9,-7,8])

    ax.plot(lle_x, lle_y, 'b.', markersize=1)


    xy = (lle_x[0], lle_y[0])
    imagebox = OffsetImage(draw_image[0], zoom=0.4, cmap=plt.cm.binary)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy, xybox=(10, 10), xycoords='data', boxcoords="offset points")
    ax.add_artist(ab)

    ax.plot(lle_x[0], lle_y[0], 'ro', markersize=2)

    now_image = np.append(now_image, 0)


    flag = True

    for j in range(1, len(x)):
        flag = True
        for item in now_image:
            if ((lle_x[item] - lle_x[j])**2 + (lle_y[item] - lle_y[j])**2) < 3e-5:
                flag = False
                break
        if(flag):
            #print(j, item)
            xy = (lle_x[j], lle_y[j])
            imagebox = OffsetImage(draw_image[j], zoom=0.4, cmap=plt.cm.binary)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, xy, xybox=(10, 10), xycoords='data', boxcoords="offset points")
            ax.add_artist(ab)

            ax.plot(lle_x[j], lle_y[j], 'ro', markersize=2)

            now_image = np.append(now_image, j)

    plt.show()