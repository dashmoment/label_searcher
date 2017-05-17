import tensorflow as tf

alpha = 0.1

def conv(input_src , weight, bias, step, size,padding='SAME'):
     
     conv = tf.nn.conv2d(input_src, weight, strides=[1, step, step, 1], padding=padding)
     conv_biased = tf.add(conv ,bias)	
     
     return tf.maximum(alpha*conv_biased,conv_biased)

 
def fc_layer(input_src, weight, bias, flat = False,linear = False):
    
    input_shape = input_src.get_shape().as_list()
    if flat:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs_transposed = tf.transpose(input_src,(0,3,1,2))
        inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
    else:
        dim = input_shape[1]
        inputs_processed = input_src
        
    if linear : return tf.add(tf.matmul(inputs_processed,weight),bias)
    
    ip = tf.add(tf.matmul(inputs_processed,weight),bias)
    return tf.maximum(alpha*ip,ip)

def pooling_layer(inputs,size,stride):
		return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME')
    
    
def model_vanilla(scopename, varscope, var_dict,inputs, ds_yolo, keep_prob):
    
     
    with tf.name_scope(scopename):
        with tf.variable_scope(varscope) as scope:
            
            scope.reuse_variables()
            varlist = var_dict[varscope]
         
            ds_yolo = {}
             
            for i in range(len(varlist)):
                 ds_yolo[varlist[i]] = tf.get_variable(varlist[i])
         
        with tf.name_scope("conv1"):
            conv1 = conv(inputs, ds_yolo['conv1w'], ds_yolo['conv1b'],1,3)
            
        with tf.name_scope("pool1"):
            pool1 = pooling_layer(conv1,3,3)
            
        with tf.name_scope("conv2"):
            conv2 = conv(pool1, ds_yolo['conv2w'], ds_yolo['conv2b'],1,3)
            
        with tf.name_scope("pool2"):
            pool2 = pooling_layer(conv2,3,3)
            
        with tf.name_scope("conv3"):
            conv3 = conv(pool2, ds_yolo['conv3w'], ds_yolo['conv3b'],1,3)
            
        with tf.name_scope("pool3"):
            pool3 = pooling_layer(conv3,3,3)
            
        with tf.name_scope("conv4"):
            conv4 = conv(pool3, ds_yolo['conv4w'], ds_yolo['conv4b'],1,3)
            
        with tf.name_scope("pool4"):
            pool4 = pooling_layer(conv4,3,3)
            
            
        with tf.name_scope("fc10"):
            fc10 = fc_layer(pool4, ds_yolo['fc10w'], ds_yolo['fc10b'],flat=True,linear=False)
            
        with tf.name_scope("dropout1"):
            dropout1 = tf.nn.dropout(fc10, keep_prob)
        
        with tf.name_scope("fc11"):
            fc11 = fc_layer(dropout1, ds_yolo['fc11w'], ds_yolo['fc11b'],flat=False,linear=False)
        
        with tf.name_scope("fc12"):
            fc12 = fc_layer(fc11, ds_yolo['fc12w'], ds_yolo['fc12b'],flat=False,linear=True)
    return fc12


def create_variable(scope, var_list):
    
    scope_dict = {}
    
    with tf.variable_scope(scope):
        
        name_dict = []
        for i in range(len(var_list)):
            
            tf.get_variable(var_list[i][0],var_list[i][1], initializer=tf.contrib.layers.xavier_initializer())
            name_dict.append(var_list[i][0])
    scope_dict[scope] = name_dict 

    return scope_dict