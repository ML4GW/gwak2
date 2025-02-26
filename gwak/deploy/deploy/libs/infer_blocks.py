import re
import h5py
import time
import logging
import psutil
import socket

import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from hermes.aeriel.client import InferenceClient



def get_ip_address() -> str:
    """
    Get the local nodes cluster-internal IP address
    """
    for _, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if (
                addr.family == socket.AF_INET
                and not addr.address.startswith("127.")
            ):
                
                return addr.address


def read_h5_data(
    test_data_dir: Path,
    key: str = "data"
):
    """
    test_data_dir: A directory that contains a list of hdf file. 
    """

    data_files= sorted(test_data_dir.glob("*.h5"))

    data_list = []
    for file in data_files: 

        with h5py.File(file, "r") as h1:
            data = h1[key][:]

        data_list.append(data.astype("float32"))

    return data_list


def static_infer_process(
    batcher,
    num_ifo, 
    psd_length,
    kernel_length,
    fduration,
    stride_batch_size,
    sample_rate, 
    client: InferenceClient=None,
    client_stream: InferenceClient=None,
    patient: float=1e-1,  
    loop_verbose: int=100
): 
    """
    The client need to connect to Triton serve already
    """
    results = []
    if client is not None: 
        segment_size = int((psd_length + kernel_length + fduration + stride_batch_size - 1) * sample_rate)

        for i, background in enumerate(batcher):

            if i % loop_verbose == 0: 
                logging.info(f"Producing inference result on {i}th iteration!")

            background = background.reshape(-1, num_ifo, segment_size)
            client.infer(background,request_id=i)

            # Wait for the Queue to return the result
            time.sleep(patient)
            result = client.get()
            while result is None:

                result = client.get()
            results.append(result[0])

        return results
    
    if client_stream is not None: 
        
        length = 1
        np.random.seed(1)
        background = np.random.normal(0, 1, (2, 2, 2048)).astype("float32")
        for i in range(length):

            
            sequence_start = (i == 0)
            sequence_end = (i == len(sequence) - 1)

            client_stream.infer(
                bh_state,
                request_id=i,
                sequence_id=8001,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
            )
                
            # Wait for the Queue to return the result
            time.sleep(patient)
            # breakpoint()
            result = client.get()
            while result is None:

                result = client.get()
                # print(f"RESULT = {result}")
            results.append(result[0])
        return results
    
    
    
def stream_jobs(
    client,
    sequence,
    sequence_id
):
    
    results = []
    with client:

        for i, (bh_state, inj_state) in enumerate(tqdm(sequence)):

            sequence_start = (i == 0)
            sequence_end = (i == len(sequence) - 1)
            
            client.infer(
                bh_state,
                request_id=i,
                sequence_id=sequence_id,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
            )

            result = client.get()
            while result is None:
                result = client.get()

            results.append(result[0])
            
    results = np.stack(results)