
# coding: utf-8

# In[ ]:

# % matplotlib inline
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import random
import glob
from skimage.feature import hog
from PIL import Image
# import cv2
from sklearn.model_selection import train_test_split


# In[ ]:

# test_file_dir = './imageData/楷书/丐/张浚张浚0.jpg'
# img = Image.open(test_file_dir).resize((100,100)).convert('L')
# a = list(img.getdata())
# b = np.reshape(a,(100,100))

# features,img = hog(b, orientations=4, pixels_per_cell=(6,6),cells_per_block=(1,1),visualise=True)
# plt.imshow(img,cmap='gray')


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
#                 fileFeaturesList.append(char_item)
                fileLabelList.append(char_styles.index(style))
            else:
                print 'there is no img under dir: ' + chars
                continue
                
    return fileFeaturesList,fileLabelList
           


# In[ ]:

# get file hog features and labels list
fileFeaturesList,fileLabelList = get_file_hog_label_list_from_disk()


# split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(fileFeaturesList, fileLabelList,
                                                    test_size=0.25, random_state=42)

# convert python list to tensor list
X_train = tf.convert_to_tensor(value=X_train,dtype=tf.float32)
X_test = tf.convert_to_tensor(value=X_test,dtype=tf.float32)

y_train = tf.convert_to_tensor(value=y_train,dtype=tf.int64)
y_test = tf.convert_to_tensor(value=y_test,dtype=tf.int64)


# one-hot encode
y_train_onehot = tf.one_hot(indices = y_train,
                            depth = 5,
                            on_value = 1,
                            off_value = 0,
                            axis = -1)
y_test_onehot = tf.one_hot(indices = y_test,
                                      depth = 5,
                                      on_value = 1,
                                      off_value = 0,
                                      axis = -1)

# put tensorlist to queues
train_input_queue = tf.train.slice_input_producer(tensor_list=[X_train,y_train_onehot],
                                                      shuffle=True)
test_input_queue = tf.train.slice_input_producer(tensor_list=[X_test,y_test_onehot],
                                                    shuffle=True)
# new image size
resize_heights = 128
resize_widths = 128

train_images,train_labels =train_input_queue[0],train_input_queue[1]

test_images,test_labels = test_input_queue[0],test_input_queue[1]

# set batch
bat_size = 100
threads_cout = 4
train_batch = tf.train.batch(tensors=[train_images,train_labels],
                             batch_size=bat_size,
                             num_threads=threads_cout)
test_batch = tf.train.batch(tensors=[test_images,test_labels],
                                 batch_size=y_test.shape[0],
                                 num_threads=threads_cout,)


# In[ ]:

# input x
x = tf.placeholder(dtype=tf.float32,shape=[None,1024],name="input_x")

# weights & biases
w = tf.Variable(tf.truncated_normal(shape=[1024,5]),name="w")
b = tf.Variable(tf.truncated_normal(shape=[5]),name="b")

tf.summary.histogram("w",w)
tf.summary.histogram("b",b)


logits = tf.add(tf.matmul(x,w,name="muti"),b,name="add")

y_true = tf.placeholder(dtype=tf.float32,shape=[None,5],name="y_true")

with tf.name_scope("cost"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true,name="cross_entropy")
    cost = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cost',cost)

with tf.name_scope("optimizer"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(cost)

with tf.name_scope("accuracy"):
    y_pred_cls = tf.arg_max(logits,1)
    y_true_cls = tf.arg_max(y_true,1)
    
    whether_equals = tf.equal(y_true_cls,y_pred_cls,name="w_equals")
    accuracy = tf.reduce_mean(tf.cast(whether_equals,dtype=tf.float32,name="cast"),name="accuracy_mean")
    
    
    tf.summary.scalar("accuracy",accuracy)
    


# In[ ]:

session = tf.Session()
init_group = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer())

# initial variables
session.run(init_group)
# merge summary
summary_merged = tf.summary.merge_all()

# train and test writer
train_writer = tf.summary.FileWriter(logdir='./log/train',graph=session.graph)
test_writer = tf.summary.FileWriter(logdir='./log/test')

# start queue
coordinator = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=session,coord=coordinator)


# In[ ]:

# iterations steps
num_iterations = 200000
x_test,y_test = session.run(test_batch)
feed_dict_test = {
    x:x_test,
    y_true:y_test
}

for i in range(num_iterations):
    x_batch,y_batch = session.run(train_batch)
    
    feed_dict_train = {
        x:x_batch,
        y_true:y_batch
    }
    
    print "start %s" % i
    if i % 5 ==0:
        # print and summary train accuracy
        s,acc = session.run([summary_merged,accuracy],feed_dict=feed_dict_train)
        train_writer.add_summary(summary=s,global_step=i)
        print "Train accuracy:{:0.1%}".format(acc)
    if i % 10 == 0:
        # print and summary test accuracy 
        s,acc = session.run([summary_merged,accuracy],feed_dict=feed_dict_test)
        test_writer.add_summary(summary=s,global_step=i)
        print "Test accuracy:{:0.1%}".format(acc)
        
        
    session.run(optimizer,feed_dict=feed_dict_train)

