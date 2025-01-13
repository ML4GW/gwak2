
from hermes.quiver.model import EnsembleModel, ExposedTensor


from hermes.quiver import Platform
from hermes.quiver.streaming import utils as streaming_utils
from hermes.quiver.model import EnsembleModel, ExposedTensor

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


class BackgroundSnapshotter(torch.nn.Module):
    """Update a kernel with a new piece of streaming data"""

    def __init__(
        self,
        psd_length,
        kernel_length,
        fduration,
        sample_rate,
        inference_sampling_rate,
    ) -> None:
        
        super().__init__()
        state_length = kernel_length + fduration + psd_length
        state_length -= 1 / inference_sampling_rate
        self.state_size = int(state_length * sample_rate)

    def forward(
            self, 
            update: Tensor, 
            snapshot: Tensor
        ) -> Tuple[Tensor, ...]:

        x = torch.cat([snapshot, update], axis=-1)
        snapshot = x[:, :, -self.state_size :]

        return x, snapshot



def add_whiten_streamer(
    ensemble: EnsembleModel,
    input: ExposedTensor,
    psd_length: float,
    sample_rate: float,
    kernel_length: float, 
    inference_sampling_rate: float,
):

    batch_size, num_ifos, *kernel_size = input.shape
    
    stride = int(sample_rate / inference_sampling_rate)
    # state_shape = (2, num_ifos, snapshotter.state_size)
    input_shape = (2, num_ifos, batch_size * stride)
    
    streaming_model = streaming_utils.add_streaming_model(
        ensemble.repository,
        # streaming_layer=snapshotter,
        name="snapshotter",
        input_name="stream",
        input_shape=input_shape,
        state_names=["snapshot"],
        # state_shapes=[state_shape],
        output_names=["strain"],
        # streams_per_gpu=streams_per_gpu,
    )