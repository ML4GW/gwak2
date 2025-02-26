import time
import h5py
import logging

import numpy as np

from tqdm import tqdm
from zlib import adler32
from pathlib import Path

from hermes.aeriel.serve import serve
from hermes.aeriel.client import InferenceClient

from libs.infer_blocks import get_ip_address, read_h5_data, static_infer_process, stream_jobs
from deploy.libs import gwak_logger
from infer_data import get_shifts_meta_data, Sequence


def infer(
    ifos: list[str],
#     psd_length: float,
#     fduration: float,
    kernel_length: float,
    Tb: int,
    batch_size: int,
#     stride_batch_size: int,
    sample_rate: int,
    fname: Path, 
    data_foramt: str,
    shifts: list[float], 
    project: str,
    model_repo_dir: Path,
    image: Path,
    result_dir: Path,
    load_model_patients: int=10,
    inference_sampling_rate: float = 1,
    **kwargs,
):
    
    kwargs={}
    result_dir = result_dir / project
    model_repo_dir = model_repo_dir / project
    
    log_file = result_dir / "log.log"
    triton_log = result_dir / "triton.log"
    result_dir.mkdir(parents=True, exist_ok=True)

    gwak_logger(log_file)
    
    ip = get_ip_address()
    kernel_size = int(kernel_length * sample_rate)
    gwak_streamer = f"gwak-{project}-streamer"

    # Data handler
    logging.info(f"Loading data files ...")
    logging.info(f"    Data directory at {fname}")
    # breakpoint()
    num_shifts, fnames, segments = get_shifts_meta_data(fname, Tb, shifts)

    address=f"{ip}:8001"
    serve_context = serve(
        model_repo_dir, 
        image, 
        log_file=triton_log, 
        wait=False
    )   

    with serve_context:
        
        for fname, (seg_start, seg_end) in zip(fnames, segments):
            for shift in range(num_shifts):
                
                fingerprint = f"{seg_start}{seg_end}{shift}".encode()
                sequence_id = adler32(fingerprint)
                
                _shifts = [s * (shift + 1) for s in shifts]

                # One segment of strain data
                sequence = Sequence(
                    fname=fname,
                    shifts=_shifts,
                    batch_size=batch_size,
                    ifos=ifos,
                    kernel_size=kernel_size,
                    sample_rate=sample_rate,
                    inference_sampling_rate=inference_sampling_rate,
                    inj_type=None,
                )

                # Wait for the serve to connect!
                logging.info(f"Waiting {load_model_patients} seconds to load model to triton!")
                time.sleep(load_model_patients)

                client_1 = InferenceClient(address, gwak_streamer)

                results = []
                with client_1:
                    logging.info(f"Inference on sequence_id: {sequence_id}")
                    for i, (bh_state, inj_state) in enumerate(tqdm(sequence)):

                        sequence_start = (i == 0)
                        sequence_end = (i == (len(sequence) - 1))

                        client_1.infer(
                            bh_state,
                            request_id=i,
                            sequence_id=sequence_id,
                            sequence_start=sequence_start,
                            sequence_end=sequence_end,
                        )

                        result = client_1.get()
                        while result is None:
                            result = client_1.get()

                        results.append(result[0])

                results = np.stack(results)

                result_file = result_dir / f"sequence_{sequence_id}.h5"
                logging.info(f"Collecting result to {result_file.resolve()}")
                time.sleep(5)
                with h5py.File(result_file, "w") as f:
                    f.create_dataset(f"data", data=results)
            