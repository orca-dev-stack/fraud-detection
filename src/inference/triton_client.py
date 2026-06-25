import numpy as np
import tritonclient.grpc as grpc
from utils.logger import get_logger

logger = get_logger("triton_client")

client = grpc.InferenceServerClient("localhost:8001")

def triton_predict(model_name, gpu_features):
    inputs = []
    outputs = []

    arr = gpu_features.to_pandas().to_numpy().astype(np.float32)

    inp = grpc.InferInput("input__0", arr.shape, "FP32")
    inp.set_data_from_numpy(arr)
    inputs.append(inp)

    out = grpc.InferRequestedOutput("output__0")
    outputs.append(out)

    result = client.infer(model_name, inputs, outputs=outputs)
    preds = result.as_numpy("output__0")
    logger.info(f"Triton inference: {preds.shape}")
    return preds
