import logging as logger
import numpy as np
import os
import onnxruntime as ort
 
 
ORT_TO_NP_TYPE = {
    "tensor(bool)": np.bool_,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int16)": np.int16,
    "tensor(uint16)": np.uint16,
    "tensor(int32)": np.int32,
    "tensor(uint32)": np.uint32,
    "tensor(int64)": np.int64,
    "tensor(uint64)": np.uint64,
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
}
 
 
class OnnxRuntimeModel:
    def __init__(self, model_path, device="cpu"):
        self.model = None
 
        providers = ["VitisAIExecutionProvider"]
        if device == "gpu":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
 
        if model_path:
            self.load_model(model_path, providers)
 
    def __call__(self, **kwargs):
        inputs = {k: np.array(v) for k, v in kwargs.items()}
        return self.model.run(None, inputs)
 
    def load_model(self, path: str, providers=None, sess_options=None):
        """
        Loads an ONNX Inference session with an ExecutionProvider. Default provider is `VitisAIExecutionProvider`
        Arguments:
            path (`str` or `Path`):
                Directory from which to load
            provider(`str`, *optional*):
                Onnxruntime execution provider to use for loading the model, defaults to `VitisAIExecutionProvider`
        """
        if providers is None:
            logger.info("No onnxruntime provider specified, using VitisAIExecutionProvider")
            #providers = ['CPUExecutionProvider']
            providers = ['VitisAIExecutionProvider']
            sess_options = [{"config_file":"C:/Users/bruce/voe-3.0-win_amd64/Install/vaip_config.json"}]

        #self.model = ort.InferenceSession(path, providers=providers)
        self.model = ort.InferenceSession(path, providers=providers, sess_options=sess_options)
