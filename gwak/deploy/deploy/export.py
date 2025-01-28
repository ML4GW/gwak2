import torch
from pathlib import Path

import hermes.quiver as qv

from deploy.libs import scale_model, add_streaming_input_preprocessor

def export(
    project: Path,
    model_dir: Path,
    triton_dir: Path,
    clean: bool,
    batch_size: int, 
    kernel_size: int, 
    num_ifos: int, 
    gwak_instances: int, 
    psd_length: float,
    # version: int,
    kernel_length: float,
    fduration: float,
    fftlength: int,
    inference_sampling_rate: int,
    sample_rate: int,
    preproc_instances: int,
    # streams_per_gpu: int,
    platform: qv.Platform = qv.Platform.ONNX,
    **kwargs,
):
    
    weights = model_dir / project / "model_JIT.pt"
    
    with open(weights, "rb") as f:
        graph = torch.jit.load(f)

    graph.eval()

    triton_dir.mkdir(parents=True, exist_ok=True)
    repo = qv.ModelRepository(triton_dir, clean=clean)

    try:
        gwak = repo.models[f"gwak-{project}"]
    except KeyError:
        gwak = repo.add(f"gwak-{project}", platform)

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

    gwak.export_version(
        graph, 
        # version=version,
        input_shapes={"whitened": input_shape}, 
        output_names=["discriminator"],
        # input_shapes={"INPUT__0": input_shape}, # TORCHSCRIPT format
        # output_names=["OUTPUT__0"],
        **kwargs,
    )

    ensemble_name = f"gwak-{project}-streamer"

    try:
        # first see if we have an existing
        # ensemble with the given name
        ensemble = repo.models[ensemble_name]
    except KeyError:
        # if we don't, create one
        ensemble = repo.add(ensemble_name, platform=qv.Platform.ENSEMBLE)

        whitened = add_streaming_input_preprocessor(
            ensemble,
            gwak.inputs["whitened"],
            psd_length=psd_length,
            sample_rate=sample_rate,
            kernel_length=kernel_length,
            inference_sampling_rate=inference_sampling_rate,
            fduration=fduration,
            fftlength=fftlength,
            preproc_instances=preproc_instances,
            # streams_per_gpu=streams_per_gpu,
        )

        ensemble.pipe(whitened, gwak.inputs["whitened"])

        # export the ensemble model, which basically amounts
        # to writing its config and creating an empty version entry
        ensemble.add_output(gwak.outputs["discriminator"])
        ensemble.export_version(None)

    else:
        # if there does already exist an ensemble by
        # the given name, make sure it has aframe
        # and the snapshotter as a part of its models
        if gwak not in ensemble.models:
            raise ValueError(
                "Ensemble model '{}' already in repository "
                "but doesn't include model 'aframe'".format(ensemble_name)
            )
        # TODO: checks for snapshotter and preprocessor

    # keep snapshot states around for a long time in case there are
    # unexpected bottlenecks which throttle update for a few seconds
    snapshotter = repo.models["snapshotter"]
    snapshotter.config.sequence_batching.max_sequence_idle_microseconds = int(
        6e10
    )
    snapshotter.config.write()