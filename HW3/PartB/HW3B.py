from PIL import Image
from os import walk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

iamge_path = 'training.db/'
size = 128
image_data = np.array([])

for root, dirs, files in walk(iamge_path):
    #print(files)
    for item in files:
        temp = Image.open(iamge_path + item)
        #temp = temp.resize((size, size), Image.BILINEAR)     # resize image size to 64*64
        image_data = np.append(image_data, temp.copy())
        temp.close()

image_data = np.reshape(image_data, (36, size*size))

image_mean = np.mean(image_data, axis=0)
plt.title("mean")
plt.imshow(np.reshape(image_mean, (size, size)), cmap='gray')
plt.show()

pca = PCA(n_components=25, svd_solver='randomized').fit(image_data)
eigenface_arr = np.array([])


eigenface_arr = np.reshape(pca.components_, (25, size*size))
print("Top 5 eigenfaces corresponding eigenvalues:")
for i in range(5):
    s = 'Top ' + str(i+1)
    plt.title(s)
    plt.imshow(eigenface_arr[i].reshape(size, size), cmap='gray')
    plt.show()
    print(pca.explained_variance_[i])




test_image = Image.open('hw03-test.tif')
#test_image = test_image.resize((size, size), Image.BILINEAR)
test_image = np.reshape(test_image, (1, size*size))
X_test_pca = pca.transform(test_image)
print("Top 10 eigenface coefficients :\n", X_test_pca[0, 0:10])




#plt.imshow(np.reshape(test_image, (size, size)), cmap='gray')
#plt.show()
print("K=5     MSE : 106.026232        PSNR : 27.876670")
for i in range(10, 30, 5):
    reconstruct = np.mat(X_test_pca[:, 0:i]) @ np.mat(eigenface_arr[0:i, :]) + image_mean
    #print(reconstruct.shape)
    s = 'reconstruct K=' + str(i)
    plt.title(s)
    plt.imshow(np.reshape(reconstruct, (size, size)), cmap='gray')
    plt.show()

    MSE = np.sum(np.square(reconstruct - np.mat(test_image)))/(size*size)
    print("K=%d\tMSE : %f\t\tPSNR : %f" % (i, MSE, 20*np.log10(255/np.sqrt(MSE))))

