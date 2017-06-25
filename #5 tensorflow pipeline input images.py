
# coding: utf-8

# # step 1. get file's name and labels list 
# # step 2. (optional) shuffle and set epochs
# # step 3. put file list into queues
# # step 4. files pre operations (like images crop)
# # step 5. set batchs (threads)

# In[1]:

# import tools
get_ipython().magic(u'matplotlib inline')
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
import random


# In[2]:

# return nohiden files dir
def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))
# read filesname and labels list
def read_list_from_disk():
    filenameslist = []
    filelabelslist = []
    # 5 styles
    styles = ['篆书','隶书','楷书','行书','草书']
    for style_item in styles:
        dir_name = './images/'+style_item
        for files in os.listdir(dir_name):
            # file name 
            filenameslist.append( dir_name+'/'+files )
            # file label
            filelabelslist.append(styles.index(style_item))
    return filenameslist,filelabelslist

# image widths and heights
IMG_WIDTHS = 100
IMG_HEIGHTS = 100

# image_operation
def image_operate(input_queue):
    
    label = input_queue[1]
    # get contents from file
    contents = tf.read_file(input_queue[0])
    # decode image by its format
    image = tf.image.decode_gif(contents)
    # resize 
    image = tf.image.resize_images(images=image,
                                   size=[IMG_WIDTHS,IMG_HEIGHTS])
    # rgb to grayscale
    image = tf.image.rgb_to_grayscale(images=image)
    
    image = tf.reshape(image,tf.stack([IMG_WIDTHS,IMG_HEIGHTS,1]))
    
    return image,label
    


# In[3]:

# get list of filename and labels
filenameslist,filelabelslist = read_list_from_disk()
# convert python list to tensor list
filenameslist_tensor = tf.convert_to_tensor(filenameslist,
                                            dtype=tf.string)
filelabelslist_tensor = tf.convert_to_tensor(filelabelslist,
                                             dtype=tf.int16)

# partition list into train and test parts
partitions = [0]*len(filelabelslist)
testfilesize = len(filelabelslist)/3
partitions[:testfilesize] = [1]*testfilesize
random.shuffle(partitions)

# dynamic partition
trainfilelist,testfilelist = tf.dynamic_partition(data=filenameslist_tensor,
                                                  partitions=partitions,
                                                  num_partitions=2)
trainlabellist,testlabellist = tf.dynamic_partition(data=filelabelslist_tensor,
                                                    partitions=partitions,
                                                    num_partitions=2)

# train and test queue
train_input_queue = tf.train.slice_input_producer(tensor_list=[trainfilelist,trainlabellist],
                                                  shuffle=False,num_epochs=2)
test_input_queue = tf.train.slice_input_producer(tensor_list=[testfilelist,testlabellist],
                                                 shuffle=False,num_epochs=2)

# images operations
trainfile , trainlabel = image_operate(train_input_queue)
testfile, testlabel = image_operate(test_input_queue)


# batch size 
BATCH_SIZE = 2

# set batch
train_batch = tf.train.batch(tensors=[trainfile , trainlabel],
                             batch_size=BATCH_SIZE,
                             num_threads=2)
test_batch = tf.train.batch(tensors=[testfile, testlabel],
                            batch_size=BATCH_SIZE,
                            num_threads=2)


# In[4]:

# session
with tf.Session() as session:
    # initial all variables
    ini_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    session.run(ini_op)
    # start queues or process will be jammed
    # set threads coordinator
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session,
                                           coord=coordinator)
    for i in range(10):
        try:
            files,label = session.run(test_batch)
            print label
        except Exception as ex:
            print type(ex).__name__ 
            
    coordinator.request_stop()
    coordinator.join(threads)
    session.close()

