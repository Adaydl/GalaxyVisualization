#%%
from __future__ import print_function, division
import tensorflow as tf

#%%
def inference(images, batch_size, n_classes,keep_prob):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]
    tf.summary.image('image',images,5)
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', 
                                  shape = [6,6,3,32],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        tf.add_to_collection('conv_weights', weights)
        
        biases = tf.get_variable('biases', 
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)
#        tf.summary.image(scope.name+'conv1', conv1,5)
        tf.add_to_collection('conv_output',conv1)
    
    #pool1 
    with tf.variable_scope('pooling1') as scope:
        
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')

        tf.add_to_collection('conv_output', pool1)
         

    
    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5,5,32,64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        
        tf.add_to_collection('conv_weights', weights)
        
        biases = tf.get_variable('biases',
                                 shape=[64], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1, weights, strides=[1,1,1,1],padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

        tf.add_to_collection('conv_output',conv2)
        
    # pool2
    with tf.variable_scope('pooling2') as scope:
       
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1],
                               padding='SAME',name='pooling2')

        tf.add_to_collection('conv_output', pool2)
        
    #conv3
    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,64,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        tf.add_to_collection('conv_weights', weights)
        biases = tf.get_variable('biases',
                                 shape=[128], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2, weights, strides=[1,1,1,1],padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name='conv3')

        tf.add_to_collection('conv_output',conv3)
    

    #conv4
    with tf.variable_scope('conv4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,128,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        tf.add_to_collection('conv_weights', weights)
        
        biases = tf.get_variable('biases',
                                 shape=[128], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv3, weights, strides=[1,1,1,1],padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name='conv4')
        tf.add_to_collection('conv_output',conv4)
    


    #pool4
    with tf.variable_scope('pooling4') as scope:
        pool4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1],
                               padding='SAME',name='pooling4')
        tf.add_to_collection('conv_output', pool4)

      
    

    #local5
    with tf.variable_scope('local5') as scope:
        reshape = tf.reshape(pool4, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,2048],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.001,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[2048],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.01))
        local5 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='local5')    

    #dropout1
    dropout1=tf.nn.dropout(local5,keep_prob)
#    dropout1 = batch_norm(local6)
     
    #local6
    with tf.variable_scope('local6') as scope:
        weights = tf.get_variable('weights',
                                  shape=[2048,2048],
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.001,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[2048],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.01))
        local6 = tf.nn.relu(tf.matmul(dropout1, weights) + biases, name='local6')
    
    #dropout2
    dropout2=tf.nn.dropout(local6,keep_prob)
#    dropout2 = batch_norm(local6)
     
    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[2048, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[n_classes],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(dropout2, weights), biases, name='softmax_linear')
    
    return softmax_linear,local6
#%%
def batch_norm(x):
    '''Batch normlization(I didn't include the offset and scale)
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x
#%%
def loss(logits, labels):
    '''Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    '''
    with tf.name_scope('Loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross-entropy')
        #logits=tf.nn.softmax(logits)
        #cross_entropy =-tf.reduce_sum(labels*tf.log(logits)+(1-labels)*tf.log(1-logits),reduction_indices=[1])
        loss = tf.reduce_mean(cross_entropy, name='loss')
        #logits=tf.nn.softmax(logits)
        #loss=0.5*tf.reduce_sum(tf.pow(tf.subtract(logits,labels),2.0))
        
        tf.add_to_collection('losses',loss)
        losses=tf.add_n( tf.get_collection('losses'),name='total_loss')
        tf.summary.scalar(scope+'/loss', losses)
        return losses


#%%
def compute_rmse(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, 
  """
  with tf.name_scope('RMSE') as scope:
      logits=tf.nn.softmax(logits,dim=-1)
#      rmse=tf.metrics.root_mean_squared_error(labels,logits)
      rmse=tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(labels-logits),reduction_indices=[1]))) #按行求和
      tf.summary.scalar(scope+'/rmse', rmse)
  return rmse  
  
#%%
def accuracy(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, 
  """
  with tf.name_scope('Accuracy') as scope:
      correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
      correct = tf.cast(correct, tf.float32)
      accuracy = tf.reduce_mean(correct)*100.0
      tf.summary.scalar(scope+'/accuracy', accuracy)
  return accuracy



#%%
def num_correct_prediction(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Return:
      the number of correct predictions
  """
  correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  correct = tf.cast(correct, tf.int32)
  n_correct = tf.reduce_sum(correct)
  return n_correct



#%%
def optimize(loss, learning_rate, global_step):
    '''optimization, use Gradient Descent as default
    '''
    with tf.name_scope('optimizer')as scope:
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,20000, 0.1, staircase=True)
        tf.summary.scalar(scope+'/learning_rate', learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')
#        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9,use_nesterov=True)
#        optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate,decay=0.9,momentum=0.0,epsilon=1e-10,use_locking=False,centered=False,name='RMSProp')
#        optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate,initial_accumulator_value=0.1,use_locking=False,name='Adagrad')
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op



