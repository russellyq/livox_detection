#from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorrt as trt
import os
import numpy as np
import tensorflow as tf
import pycuda.driver as cuda
import pycuda.autoinit
# import keras.backend.tensorflow_backend as KTF

def do_inference_trt(context, bindings, inputs, outputs, stream):
    # Transfer data from CPU to the GPU.
    # [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
    cuda.memcpy_htod_async(inputs.device, inputs.host, stream)

    # # Run inference.
    context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
        
        
    # # Transfer predictions back from the GPU.
    # [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
    cuda.memcpy_dtoh_async(outputs.host, outputs.device, stream)

    # # Synchronize the stream
    stream.synchronize()

    return outputs.host
        # return [out.host for out in self.outputs]

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTModel(object):
    def __init__(self, input_shape, output_shape) -> None:
        super().__init__()
        self.gpu_config()
        self.load_trt_model()
        self.inputs, self.outputs, self.bindings, self.stream =self.allocate_buffers(input_shape, output_shape)

    def gpu_config(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        # config.allow_soft_placement = True
        # config.log_device_placement = False
        sess = tf.Session(config=config)
        # KTF.set_session(sess)
        self.sess = sess
    
    def load_trt_model(self):
        logger = trt.Logger(trt.Logger.INFO)
        with open("./model_pb/saved_engine.trt", "rb") as f, trt.Runtime(logger) as runtime:
            self.engine=runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
    
    def allocate_buffers(self, input_shape, output_shape):

        stream = cuda.Stream()

        input_host_mem = cuda.pagelocked_empty(input_shape, dtype=trt.nptype(trt.float32))
        out_host_mem = cuda.pagelocked_empty(output_shape, dtype=trt.nptype(trt.float32))
        
        input_device_mem = cuda.mem_alloc(input_host_mem.nbytes)
        output_device_mem = cuda.mem_alloc(out_host_mem.nbytes)
        
        bindings = [int(input_device_mem), int(output_device_mem)]

        inputs = HostDeviceMem(input_host_mem, input_device_mem)
        outputs = HostDeviceMem(out_host_mem, output_device_mem)
        return inputs, outputs, bindings, stream
    
    def detect(self, batch_bev_img):
        self.inputs.host = batch_bev_img
        trt_outputs = do_inference_trt(self.context, self.bindings, self.inputs, self.outputs, self.stream)
        return trt_outputs
