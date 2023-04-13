import tensorrt as trt
import argparse
from onnx import ModelProto
import os
import sys
import onnx
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
import onnxruntime as rt
# def build_engine(onnx_path):

#    """
#    This is the function to create the TensorRT engine
#    Args:
#       onnx_path : Path to onnx_file. 
#       shape : Shape of the input of the ONNX file. 
#   """
#    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
#        #builder.max_workspace_size = (256 << 20)
#        with open(onnx_path, 'rb') as model:
#            parser.parse(model.read())
#        engine = builder.build_engine(network)
#        return engine
def build_engine(onnx_file_path):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            #builder.max_workspace_size = 1 << 28 # 256MiB
            #builder.max_batch_size = 1
            # Parse model file
            
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_engine(network)
            print("Completed creating Engine")

            return engine

def save_engine(engine, file_name):
   buf = engine.serialize()
   with open(file_name, 'wb') as f:
       f.write(buf)

def load_engine(trt_runtime, engine_path):
   with open(engine_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine





# onnx_path = "./model_pb/save_model.onnx"
# # trt_path = './model_pb/save_model.trt'

# onnx_model = onnx.load(onnx_path)
# onnx.checker.check_model(onnx_model)


# engine = build_engine(onnx_path)



logger = trt.Logger(trt.Logger.INFO)
with open("./model_pb/saved_engine.trt", "rb") as f, trt.Runtime(logger) as runtime:
    engine=runtime.deserialize_cuda_engine(f.read())
# Create the context for this engine
# context = engine.create_execution_context()

for idx in range(engine.num_bindings):
    is_input = engine.binding_is_input(idx)
    name = engine.get_binding_name(idx)
    op_type = engine.get_binding_dtype(idx)
    #model_all_names.append(name)
    shape = engine.get_binding_shape(idx)
    print('input id:',idx,'   is input: ', is_input,'  binding name:', name, '  shape:', shape, 'type: ', op_type)

