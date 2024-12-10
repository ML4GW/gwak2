import torch

import hermes.quiver as qv

from hermes.aeriel.serve import serve
from tritonclient import grpc as triton

# Arguments

# -----
gwak_instances = 6
# -----

# -----
weights = "/home/hongyin.chen/anti-gravity/gwak2/gwak/output/background/model_JIT.pt"
ept_repo = "/home/hongyin.chen/Xperimental/GWAK/export_repos"
batch_size, num_ifos, kernel_size = 1, 2, 200

clean = True # Would like to swith to False later on. 
platform = qv.Platform.TENSORRT
# -----

# Support function
def scale_model(model, instances):
    """
    Scale the model to the number of instances per GPU desired
    at inference time
    """
    # TODO: should quiver handle this under the hood?
    try:
        model.config.scale_instance_group(instances)
    except ValueError:
        model.config.add_instance_group(count=instances)

# Main function
with open(weights, "rb") as f:
    graph = nn = torch.jit.load(f)

graph.eval()

repo = qv.ModelRepository(ept_repo, clean=clean)

try:
    gwak = repo.models["gwak"]
except KeyError:
    gwak = repo.add("gwak", qv.Platform.TENSORRT)

if gwak_instances is not None:
    scale_model(gwak, gwak_instances)

input_shape = (batch_size, kernel_size, num_ifos)

kwargs = {}
if platform == qv.Platform.ONNX:
    kwargs["opset_version"] = 13

    # turn off graph optimization because of this error
    # https://github.com/triton-inference-server/server/issues/3418
    gwak.config.optimization.graph.level = -1
elif platform == qv.Platform.TENSORRT:
    kwargs["use_fp16"] = False
    kwargs["opset_version"] = 13
    
gwak.export_version(
    graph, 
    input_shapes={"some_name":input_shape}, 
    output_names=["classifier"]
)

# Build Whiten model 