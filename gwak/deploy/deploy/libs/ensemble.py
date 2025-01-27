import torch 
from typing import Callable, Optional, Tuple

from hermes.quiver.model import EnsembleModel, ExposedTensor


from hermes.quiver import Platform
from hermes.quiver.streaming import utils as streaming_utils
from hermes.quiver.model import EnsembleModel, ExposedTensor


Tensor = torch.Tensor
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