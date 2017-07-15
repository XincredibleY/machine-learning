
# coding: utf-8

# In[ ]:

# % matplotlib inline
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import random
import glob
import time


# In[ ]:

# # check our data
# test_file_dir = './imageData//楷书/丐/张浚张浚0.jpg'
# test_img = plt.imread(test_file_dir)
# print "test image shape is : \n"
# print test_img.shape
# plt.imshow(test_img)


# In[ ]:

# return nohiden files dir
def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

# get file name and label list from disk
def readFileList():
    char_styles = ['篆书','隶书','楷书','行书','草书']
    # fileNamesList and fileLabelList
    fileNameList = []
    fileLabelList = []
    # iterate all styles
    for style in char_styles:
        print 'start iterate: %s'% style
        # iterate all chars under this style
        for chars in listdir_nohidden('./imageData/'+ style):
            # there is at least one item 
            for font in listdir_nohidden(chars):
                if len(listdir_nohidden(chars)) > 0:
                # just get the first font images under this chars
                char_item =  listdir_nohidden(chars)[0]                
                print 'saving : ' + char_item
                fileNameList.append(char_item)
                fileLabelList.append(char_styles.index(style))
            else:
                print 'there is no img under dir: ' + chars
                continue
                
    return fileNameList,fileLabelList
               


# In[ ]:

# image operation
def image_operation(input_queue,grayscale,heights,weights):
    label = input_queue[1]
    contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(contents)
    # resize
    image = tf.image.resize_images(image,[heights,weights])
    # grayscale
    image = tf.image.rgb_to_grayscale(image) if grayscale else image
    image=tf.reshape(image,tf.stack([1,heights,weights]))
    return image,label


# In[ ]:

# we will build ConvNet in the following step

# define convolutional layer 
def conv_layer(name,input_tensor,filter_h,filter_w,input_channels,output_channels,use_relu=True):
    with tf.name_scope(name):
        # conv,input,filter,strides,padding,
        weight = tf.Variable(tf.truncated_normal(shape=[filter_h,filter_w,input_channels,output_channels]),name="conv_w")
        conv = tf.nn.conv2d(
            input=input_tensor,
            filter=weight,
            strides=[1,1,1,1],
            padding="SAME",
            name="conv_op",
            data_format="NCHW"
        )
        biase = tf.Variable(tf.truncated_normal(shape=[output_channels]),name="conv_b")
        
        tf.summary.histogram('conv_w',weight)
        tf.summary.histogram('conv_b',biase)
        
        conv = tf.nn.bias_add(conv,biase,name="conv_add",data_format="NCHW")
        conv = tf.nn.relu(conv,name="conv_relu") if use_relu else conv
        
        return conv
        
# define pool layer
def pool_layer(name,input_value):
    with tf.name_scope(name):
        pool = tf.nn.max_pool(
            value = input_value,
            ksize = [1,1,2,2],
            strides = [1,1,2,2],
            padding="SAME",
            name = "pool_op",
            data_format="NCHW"            
        )
        return pool
    
# define fully connected lay
def fl_layer(name,input_tensor,input_channels,output_channels):
    with tf.name_scope(name):
        weight = tf.Variable(tf.truncated_normal(shape=[input_channels,output_channels]),name="fl_w")
        biase = tf.Variable(tf.truncated_normal(shape=[output_channels]),name="fl_b")
        fl =  tf.add(tf.matmul(input_tensor,weight),biase)
        
        tf.summary.histogram('fl_w',weight)
        tf.summary.histogram('fl_b',biase)        
        return fl
# generate train and test batch    
def generate_batch(fileNameList,fileLabelList):
    with tf.name_scope('convert_to_tensor'):
        # convert python list to tensor list
        fileNameList_tensor = tf.convert_to_tensor(value=fileNameList,
                                               dtype=tf.string)
        fileLabelList_tensor = tf.convert_to_tensor(value=fileLabelList,
                                                dtype=tf.int64)
    with tf.name_scope('label_one_hot_encoding'):
        # one-hot encode
        fileLabelList_tensor = tf.one_hot(indices = fileLabelList_tensor,
                                      depth = 5,
                                      on_value = 1,
                                      off_value = 0,
                                      axis = -1)
    with tf.name_scope('dynamic_partition'):
        # dynamic partition
        trainFnameList,testFnameList = tf.dynamic_partition(data=fileNameList_tensor,
                                                        partitions=partitions,
                                                        num_partitions=2)
        trainLabelList,testLabelList = tf.dynamic_partition(data=fileLabelList_tensor,
                                                        partitions=partitions,
                                                        num_partitions=2)    
        
    with tf.name_scope('set_queues'):
        # put tensorlist to queues
        train_input_queue = tf.train.slice_input_producer(tensor_list=[trainFnameList,trainLabelList],
                                                      shuffle=True)
        test_input_queue = tf.train.slice_input_producer(tensor_list=[testFnameList,testLabelList],
                                                    shuffle=True)

    with tf.name_scope('image_operation'):
        # new image size
        resize_heights = 128
        resize_widths = 128
        train_images,train_labels = image_operation(train_input_queue,
                                                grayscale=True,
                                                heights=resize_heights,
                                                weights = resize_widths)
        test_images,test_labels = image_operation(test_input_queue,
                                              grayscale=True,
                                              heights=resize_heights,
                                              weights = resize_widths)
    with tf.name_scope('set_batch'):
        # set batch
        bat_size = 100
        threads_cout = 4
        train_batch = tf.train.batch(tensors=[train_images,train_labels],
                                 batch_size=bat_size,
                                 num_threads=threads_cout,)
        test_batch = tf.train.batch(tensors=[test_images,test_labels],
                                 batch_size=500,
                                 num_threads=threads_cout,)
    return  train_batch,test_batch
    


# In[ ]:

# main model

def cnn_model(lr=1e-3,fileNameList=None,fileLabelList=None,partitions=None,iteration=3000):
    
    
    # reset graph
    tf.reset_default_graph()
    
    resize_heights = 128
    resize_widths = 128
    
    train_batch,test_batch = generate_batch(fileNameList,fileLabelList)
    ##********************************************************************************************


    # input x , [-1,128,128,1]
    x = tf.placeholder(dtype=tf.float32,
                       shape=[None,1,resize_heights,resize_widths],
                       name='input_x')
    # must use transpose
    tf.summary.image("input_images",tensor=tf.transpose(a=x[:3],perm=[0,2,3,1]))
    # one-hot encoded labels 
    y_true = tf.placeholder(dtype=tf.int16,shape=[None,5])
    
    
    # conv 1 , [-1,128,128,32]
    conv_1 = conv_layer("conv_1",x,3,3,1,32,use_relu=True)
    # pool 1 , [-1,64,64,32]    
    pool_1 = pool_layer("pool_1",conv_1)
    # conv 2 , [-1,64,64,64]    
    conv_2 = conv_layer("conv_2",pool_1,5,5,32,64,use_relu=True)
    # pool 2 , [-1,32,32,64]    
    pool_2 = pool_layer("pool_2",conv_2)
    # conv_3 , [-1,32,32,128]
    conv_3 = conv_layer("conv_3",pool_2,7,7,64,128,use_relu=True)
    # pool 3 , [-1,16,16,128]    
    pool_3 = pool_layer("pool_3",conv_3)
    
    # conv_4 , [-1,32,32,128]
    conv_4 = conv_layer("conv_4",pool_3,9,9,128,256,use_relu=True)
    # pool 4 , [-1,16,16,128]    
    pool_4 = pool_layer("pool_4",conv_4)
    
#     # conv_5 , [-1,32,32,128]
#     conv_5 = conv_layer("conv_5",pool_4,11,11,256,512,use_relu=True)
#     # pool 5 , [-1,16,16,128]    
#     pool_5 = pool_layer("pool_5",conv_5)

      
    # flat layer , [-1,16*16*128]
    flat_layer = tf.reshape(pool_4,shape=[-1,8*8*256])
    # fl_1 , [-1,128]
    fl_1 = fl_layer("fl_1",flat_layer,8*8*256,128)
    # fl_2 , [-1,5]    
    fl_2 = fl_layer("fl_2",fl_1,128,5)
        
    
    with tf.name_scope("cost"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fl_2,labels=y_true,name="cross_entropy")
        cost = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cost',cost)
        
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr,name="AdamOptimizer").minimize(cost)
    
    with tf.name_scope("accuracy"):
        y_pred_cls = tf.arg_max(fl_2,1)
        y_true_cls = tf.arg_max(y_true,1)        
        
        whether_equals  = tf.equal(y_pred_cls,y_true_cls)
        # must set float32
        accuracy = tf.reduce_mean(tf.cast(whether_equals,dtype=tf.float32)) 
        tf.summary.scalar('cost',accuracy)        

        

    session = tf.Session()

    # initial all variables
    ini_group = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
    session.run(ini_group)
    
    summary_merge = tf.summary.merge_all()
    
    # write graph and summaries to train writer
    train_writer = tf.summary.FileWriter('./op_log/train_lr_'+str(lr) ,graph=session.graph)
    # writw summaries to test writer
    test_writer = tf.summary.FileWriter('./op_log/test_lr_'+str(lr) )
    

    # start queues
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session,
                                           coord=coordinator)
    test_x,test_y = session.run(test_batch)
    feed_dict_test = {
        x : test_x,
        y_true:test_y
    }

    with tf.name_scope("train"):
        for i in range(iteration):
            train_x,train_y = session.run(train_batch)
            feed_dict_train = {
                x : train_x,
                y_true:train_y
            }
            print 'training step:'+str(i)
            if i % 5 == 0:
                acc,sm = session.run([accuracy,summary_merge],feed_dict=feed_dict_train)
                train_writer.add_summary(sm,i)
                temp_time = time.time()
                tem_str =  ', time used : %0.2f s'  % (temp_time - start_time)
                print 'train accuracy:{:0.1%}' . format(acc) + tem_str
                
            if i % 100 == 0:
                acc,sm = session.run([accuracy,summary_merge],feed_dict=feed_dict_test)
                test_writer.add_summary(sm,i)
                temp_time = time.time()
                tem_str =  ', time used : %0.2f s'  % (temp_time - start_time)
                print "Test acc:{:0.1%}".format(acc)+ tem_str
            
            session.run(optimizer,feed_dict=feed_dict_train)
        end_time = time.time()
        print 'total time : %0.2f s'  % (end_time - start_time)



# In[ ]:

# get file names and labels list
fileNameList,fileLabelList = readFileList()

# partitions all data to train and test parts
all_data_counts = len(fileLabelList)
partitions = [0]*all_data_counts

# split all data (15585) into 3/4(11689) train data , and 1/4 (3896) test data
test_size = all_data_counts/4
partitions[:test_size] = [1]*test_size
# must shuffle
random.shuffle(partitions)


print 'all data is %s,train data is %s ,test data is %s' % (all_data_counts,(all_data_counts-test_size),test_size)

start_time = time.time()

for lr in [1e-3]:
    cnn_model(lr,fileNameList,fileLabelList,partitions,iteration=50000)

