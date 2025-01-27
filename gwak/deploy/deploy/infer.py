import time
import h5py
import logging

import numpy as np

from pathlib import Path

from hermes.aeriel.serve import serve
from hermes.aeriel.client import InferenceClient

from libs.infer_blocks import get_ip_address, infer_process


def infer(
    model_repo_dir: Path,
    image: Path,
    log_file: Path,
    test_data_dir: Path,
    infer_file: Path,
    project: str, 
    batch_size = 1,
    **kwargs,
):

    infer_file.parents[1].mkdir(parents=True, exist_ok=True)

    ip = get_ip_address()
    
    address=f"{ip}:8001"
    model_name = f"gwak-{project}"

    serve_context = serve(
        model_repo_dir, 
        image, 
        log_file=log_file, 
        wait=False
    )

    with serve_context:

        # Wait for the serve to connect!
        time.sleep(5)

        client = InferenceClient(address, model_name)

        # This part would have to replace with the Timeslide Generator
        batcher = np.random.normal(0, 1, (30, 200, 2))
        batcher = batcher.astype("float32")
        batcher = batcher.reshape((-1, batch_size, 200, 2))
        
        # Inference on Triton and return the result 
        results = infer_process(
            client,
            batcher = batcher
        )


    with h5py.File(infer_file, "w") as h:

        h.create_dataset(f"{project}", data=results)

