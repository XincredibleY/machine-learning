
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import random
import glob
from skimage.feature import hog
from PIL import Image
# import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
import itertools


# In[ ]:

test_file_dir = './imageData/行书/齐/敬世江敬世江0.jpg'
img = Image.open(test_file_dir).resize((200,200)).convert('L')
a = list(img.getdata())
b = np.reshape(a,(200,200))

features,img_h = hog(b, orientations=4, pixels_per_cell=(6,6),cells_per_block=(1,1),visualise=True)

plt.axis('off')
plt.imshow(img_h, cmap=plt.cm.gray)
plt.show()


# In[20]:

image = Image.open(test_file_dir).resize((200,200)).convert('L')

features, hog_image = hog(image, orientations=4, pixels_per_cell=(6,6),cells_per_block=(1,1),visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()


# In[ ]:

# return nohiden files dir
def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

# get hog features
def image_reader(file_name,new_size):
    # read images
    # resize it 
    # set to grayscale
    # (optional) get hog features
    img = Image.open(file_name).resize(new_size).convert('L')
    img = list(img.getdata())
    img = np.reshape(img,(100,100))
    return img

# get file name and label list from disk
def get_file_hog_label_list_from_disk():
    char_styles = ['篆书','隶书','楷书','行书','草书']
    # fileNamesList and fileLabelList
    fileFeaturesList = []
    fileLabelList = []
    # iterate all styles
    for style in char_styles:
        print 'start iterate: %s'% style
        # iterate all chars under this style
        for chars in listdir_nohidden('./imageData/'+ style):
            # there is at least one item 
            if len(listdir_nohidden(chars)) > 0:
                # just get the first font images under this chars
                char_item =  listdir_nohidden(chars)[0]                
                img = image_reader(char_item,(100,100))
                features = hog(img, orientations=4, pixels_per_cell=(6,6),cells_per_block=(1,1))
                features = list(features)
                print 'saving : ' + char_item
                fileFeaturesList.append(features)
                # fileFeaturesList.append(char_item)
                fileLabelList.append(char_styles.index(style))
            else:
                print 'there is no img under dir: ' + chars
                continue
                
    return fileFeaturesList,fileLabelList
           


# In[ ]:

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# # train the model
# # frist we use svm


# In[ ]:

fileFeaturesList,fileLabelList = get_file_hog_label_list_from_disk()
x_train,x_test,y_train,y_test  = train_test_split(fileFeaturesList,fileLabelList,
                                                  test_size=0.25,random_state=42)



cls = svm.SVC()
cls.fit(x_train,y_train)
predictLabels = cls.predict(x_test)

print  "svm acc:%s" % accuracy_score(y_test,predictLabels)


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train,y_train)
predictLabels = neigh.predict(x_test)
print  "KNN acc:%s" % accuracy_score(y_test,predictLabels)



# In[ ]:



