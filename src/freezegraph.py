import tensorflow as tf
import os
import sys
# import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile
import tensorflow as tf
import uff
# import tensorrt as trt


sys.path.append("..") 
from networks.model import *
import config.config as cfg
import numpy as np

DX = cfg.VOXEL_SIZE[0]
DY = cfg.VOXEL_SIZE[1]
DZ = cfg.VOXEL_SIZE[2]

X_MIN = cfg.RANGE['X_MIN']
X_MAX = cfg.RANGE['X_MAX']

Y_MIN = cfg.RANGE['Y_MIN']
Y_MAX = cfg.RANGE['Y_MAX']

Z_MIN = cfg.RANGE['Z_MIN']
Z_MAX = cfg.RANGE['Z_MAX']

overlap = cfg.OVERLAP
HEIGHT = round((X_MAX - X_MIN+2*overlap) / DX)
WIDTH = round((Y_MAX - Y_MIN) / DY)
CHANNELS = round((Z_MAX - Z_MIN) / DZ)



net = livox_model(HEIGHT, WIDTH, CHANNELS)
output_node_names = []
with open('./nodes.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split('\n')
        output_node_names.append(str(line[0]))
print(output_node_names)
def checkpoints2pb():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(cfg.GPU_INDEX)):
            input_bev_img_pl = net.placeholder_inputs(cfg.BATCH_SIZE)
            end_points = net.get_model(input_bev_img_pl)

            saver = tf.train.Saver()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            saver.restore(sess, "../model/livoxmodel")
            ops = {'input_bev_img_pl': input_bev_img_pl,  # input
                        'end_points': end_points,  # output
                                }

            # file=open('./nodes.txt','a+')
            # for n in tf.get_default_graph().as_graph_def().node:
            #     file.write(n.name + '\n')
            # file.close()

            output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
            # print(output_node_names)

            #output_node_names = ["Conv_39/BiasAdd"]

            # Freeze the graph
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names)
            with open('../model_pb/output_graph.pb', 'wb') as f:
                f.write(frozen_graph_def.SerializeToString())



def get_trt_graph(batch_size=cfg.BATCH_SIZE,workspace_size=1<<30):
    # conver pb to FP32pb
    with gfile.FastGFile('../model_pb/output_graph.pb','rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())


    precision_mode = "FP32" #"FP16"
    trt_graph = trt.create_inference_graph(input_graph_def=graph_def, outputs=output_node_names,
                                                max_batch_size=batch_size,
                                                max_workspace_size_bytes=workspace_size,
                                                precision_mode=precision_mode)  # Get optimized graph

    
    return trt_graph

def freeze_graph_test(pb_path='../model_pb/output_graph.pb'):
    '''#
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            converter = trt.TrtGraphConverter(
            input_graph_def=output_graph_def,
            nodes_blacklist=output_node_names) #output nodes
        trt_graph = converter.convert()
        # Import the TensorRT graph into a new graph and run:
        output_node = tf.import_graph_def(
            trt_graph,
            return_elements=output_node_names)
        #     sess.run(tf.global_variables_initializer())
 
def pb2uff():
    MODEL_DIR = '../model_pb/output_graph.pb'

    ENGINE_PATH = '../model_pb/output_graph_.pb.plan'
    INPUT_NODE = 'Placeholder'
    OUTPUT_NODE = output_node_names
    INPUT_SIZE = [1, HEIGHT, WIDTH, CHANNELS] 
    MAX_BATCH_SIZE = 1 
    MAX_WORKSPACE = 1<<30

    G_LOGGER = trt.Logger(trt.Logger.INFO)
    uff_model = uff.from_tensorflow_frozen_model(MODEL_DIR, OUTPUT_NODE, '../model_pb/model.uff')
    #parser = uffparser.create_uff_parser()
    parser = trt.UffParser()
    parser.register_input(INPUT_NODE, INPUT_SIZE, 0)
    parser.register_output(OUTPUT_NODE)
    engine = trt.utils.uff_to_trt_engine(
            G_LOGGER,
            uff_model,
            parser,
            MAX_BATCH_SIZE,
            MAX_WORKSPACE,
            datatype=trt.infer.DataType.FLOAT)
    trt.utils.cwrite_engine_to_file(ENGINE_PATH,engine.serialize())
#checkpoints2pb()
#pb2uff() 
freeze_graph_test()
#get_trt_graph()