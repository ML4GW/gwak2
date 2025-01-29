import time
import h5py
import logging

import numpy as np

from pathlib import Path

from hermes.aeriel.serve import serve
from hermes.aeriel.client import InferenceClient

from libs.infer_blocks import get_ip_address, static_infer_process


def infer(
    num_ifos: int,
    kernel_size: int,
    model_repo_dir: Path,
    image: Path,
    log_file: Path,
    test_data_dir: Path,
    infer_file: Path,
    project: str, 
    batch_size = 1,
    **kwargs,
):

    infer_file.parents[0].mkdir(parents=True, exist_ok=True)
    log_file.parents[0].mkdir(parents=True, exist_ok=True)

    ip = get_ip_address()
    
    address=f"{ip}:8001"
    # model_name = f"gwak-{project}-streamer"
    # model_1 = f"preprocessor"
    model_2 = f"gwak-{project}"
    model_2 = "gwak-white_noise_burst"
    serve_context = serve(
        model_repo_dir, 
        image, 
        log_file=log_file, 
        wait=False
    )

    with serve_context:

        # Wait for the serve to connect!
        time.sleep(5)
        # breakpoint()
        client_1 = InferenceClient(address, model_2)
        # client_2 = InferenceClient(address, model_2)

        # This part would have to replace with the Timeslide Generator
        # batcher = np.random.normal(0, 1, (3, 2, num_ifos, 137288)).astype("float32")
        batcher = np.random.normal(0, 1, (3, 2, num_ifos, 200)).astype("float32")
        # batcher = batcher.reshape((-1, batch_size, 2, 200, 2))
        # breakpoint()
        # (2, 2, 137288)
        stream_batcher = np.random.normal(0, 1, (3, 2, 2, 4096)).astype("float32")
        # Inference on Triton and return the result 
        results_1 = static_infer_process(
            batcher,
            client_1,
            # client_2,
            # batcher = batcher,
            # batcher = stream_batcher
        )


    with h5py.File(infer_file, "w") as h:
        print(results_1)
        h.create_dataset(f"{project}", data=results_1)

