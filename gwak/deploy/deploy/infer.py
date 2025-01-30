import time
import h5py
import logging

import numpy as np

from pathlib import Path

from hermes.aeriel.serve import serve
from hermes.aeriel.client import InferenceClient

from libs.infer_blocks import get_ip_address, read_h5_data, static_infer_process


def infer(
    num_ifos: int,
    psd_length: float,
    kernel_length: float,
    batch_size: int,
    sample_rate: int,
    test_data_dir: Path, 
    project:Path,
    model_repo_dir: Path,
    image: Path,
    result_dir: Path,
    **kwargs,
):


    result_dir = result_dir / project
    model_repo_dir = model_repo_dir / project
    result_dir.mkdir(parents=True, exist_ok=True)
    log_file = result_dir / "log.log"
    result_file = result_dir / "result.h5"

    ip = get_ip_address()
    address=f"{ip}:8001"
    
    whiten_model = "preprocessor"
    gwak_model = f"gwak-{project}"

    serve_context = serve(
        model_repo_dir, 
        image, 
        log_file=log_file, 
        wait=False
    )

    data_list = read_h5_data(
        test_data_dir = test_data_dir, 
        key="data"
    )

    with serve_context:

        # Wait for the serve to connect!
        logging.info("Waiting 10 seconds to load model to triton!")
        time.sleep(10)

        client_1 = InferenceClient(address, whiten_model)
        client_2 = InferenceClient(address, gwak_model)

        # Produce whiten strain
        whitened_data = static_infer_process(
            batcher=data_list,
            num_ifo=num_ifos, 
            psd_length=psd_length,
            kernel_length=kernel_length,
            batch_size=batch_size,
            sample_rate=sample_rate, 
            client=client_1,
            patient=3
        )

        # Produce inference result
        inference_result = static_infer_process(
            batcher=whitened_data,
            num_ifo=num_ifos, 
            psd_length=0,
            kernel_length=kernel_length,
            batch_size=0,
            sample_rate=sample_rate, 
            client=client_2
        )
    inference_results = np.stack(inference_result)

    with h5py.File(result_file, "w") as h:
        h.create_dataset(f"{project}", data=inference_results)

