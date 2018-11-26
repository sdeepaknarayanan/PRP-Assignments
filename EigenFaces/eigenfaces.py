import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.model_selection import train_test_split


"""
This function is used to convert the 
multi-dimensional image input into a long vector.
"""

def imagevector(image):
    vect_image = np.ravel(image)
    return vect_image
    
"""
PCA is the function used to find the Eigenvectors,
Eigenvalues of the matrix AtA, and matrix A.
"""

def pca(image_set,shape):
    
    avg_image = np.zeros(len(image_set[0]))
    
    """
    Computing the Image Psi, using all the
    images in the training set. 
    """
    
    for image in image_set:
        avg_image+=image
    avg_image = avg_image/len(image_set)
    average = avg_image
    average = average.astype(np.uint8)
    
    
    """
    The Phi's are being computed below
    """
    
    image_mean = []
    for i in range(len(image_set)):
        image_mean.append(image_set[i] - average)
        
    """
    The matrix A is being computed below
    """
    
    A = np.zeros((len(image_set[0]),len(image_set)))
    for i in range(len(image_set)):
        for j in range(len(image_set[0])):
            A[j][i] = image_mean[i][j]
    
    """
    Computation of the Eigenvalues and the Eigenvectors.
    """
    
    mat = np.transpose(A).dot(A)
    eigenvalues, eigenvectors = np.linalg.eig(mat)
    image_mean = np.asarray(image_mean)
    
    return eigenvalues, eigenvectors, A, average,image_mean
    
"""
Computing the Eigenfaces in the function below;
It is simply a normalized version of the eigenvectors
of the original AAt matrix.
"""

def eigenfaces(A,eigenvectors):
    eigenfaces = []
    for i in range(len(eigenvectors)):
        temp = A.dot(eigenvectors[i])
        temp = temp/temp.sum()
        eigenfaces.append(temp)
    eigenfaces = np.asarray(eigenfaces)
    return eigenfaces

"""
Face Recognition Main Block. 
"""
def face_recognition(test, omegas, average, actual, vect_matrix):
    count = 0
    for index in range(len(test)):
        image = x_test[index] - average
        image = np.transpose(vect_matrix).dot(image)
        min_ = np.inf
        label = 0
        for i in range(len(omegas)):
            curr = (np.linalg.norm(image/image.sum() - omegas[i]/omegas[i].sum()))
            if curr<min_:
                min_ = curr
                label = i+1
        if actual[index]==label:
            #print(test[index], index)
            count+=1
    return count/len(test)
    
# ------------------------------ Main ------------------

"""
    We are choosing the Images for Training and Testing below
"""

print(' Reading Images right now .....')
dataset = []
iter_ = 0
for i in range(1,41):
    read_dir =  'orl_faces/s'+ str(i)
    list_of_images = (os.listdir(read_dir))
    for elem in list_of_images:
        curr_image = cv.imread(read_dir+'/'+elem)
        image = curr_image
        curr_image = cv.cvtColor(curr_image, cv.COLOR_BGR2GRAY)
        curr_image = imagevector(curr_image)
        dataset.append(curr_image)
        iter_+=1
print(' All the images have been read.....')
fin = []
for i in range(len(dataset)):
    label = (i)//10+1
    temp = dataset[i]
    label = np.array(label)
    new_arr = np.array([temp,label])
    fin.append(new_arr)
print(' Splitting (Randomly) into Train and Test')
fin = np.asarray(fin)
x_train = []
x_test = []
y = fin[:,1]
x = fin[:,0]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)
print(' Splitting done .....')

"""
    Dataset has been created by now. Now we are sorting the eigenvalues 
    and the corresponding eigenvectors.
"""
print(' PCA under progess now. ')
vals, vectors, A, average,train_mean = pca(x_train,image.shape)
idx = vals.argsort()[::-1]   
vals = vals[idx]
vectors = vectors[:,idx]
#train_mean = train_mean.transpose()


vect_matrix = np.zeros((len(x_train[0]),len(x_train)))
#print(A.shape,vect_matrix.shape)
for i in range(len(vectors)):
    temp = A.dot(vectors[i])
    vect_matrix[:,i] = temp
    vect_matrix[:,i] = vect_matrix[:,i]/vect_matrix[:,i].sum()
#top_k = vect_matrix[:,:20]
#print(top_k.shape)
print(' PCA Done.')
"""
    We are creating the Omegas now.
    These will be used while assigning the face class.
    W is the weight matrix consisting of all the weights we need.
"""

print(" The average image is ")
temp = average.reshape((112,92))
plt.title(' Average Image ')
plt.imshow(temp,cmap = 'gray')
plt.show()
plt.close()
print(' A few eigenfaces are ....')
for i in range(5):
	temp = vect_matrix[:,i]
	#print(temp.shape)
	temp = temp.reshape((112,92))
	plt.title(' Eigenface '+ str(i+1))
	plt.imshow(temp, cmap = 'gray')
	plt.show()
	plt.close()
important = vect_matrix[:,:15]
#print(important.shape)
w = []
for i in range(len(x_train)):
    w.append(np.transpose(important).dot(x_train[i]))
omegas = []
for i in range(0,280,7):
	omegas.append(w[i])

image = x_test[0] - average
image = np.transpose(vect_matrix).dot(image)
#print(len(omegas),len(w))

print(' Accuracy is ',(face_recognition(x_test,omegas,average,y_test,important)*100))



def isimage(image,eigenfaces,average):
    image = imagevector(image)
    image = image- average
    vect = eigenfaces.dot(image)
    if np.linalg.norm(vect - image)>=3500:
        print(' Not a face')
        return 0
    else:
        print(' It is a face')
        return 1


    
