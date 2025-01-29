import psutil
import socket

from pathlib import Path

from hermes.aeriel.client import InferenceClient


import logging
import time
from dataclasses import dataclass

import numpy as np


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


def static_infer_process(
    batcher,
    client_1: InferenceClient,
    # client_2: InferenceClient,
    loop_verbose: int=100
): 
    """
    The client need to connect to Triton serve already
    """
    # results_1 = []
    results_2 = []
    for i, background in enumerate(batcher):
        print(i)
        if i % loop_verbose == 0: 
            logging.info(f"Producing inference on {i}th iteration!")
        background = np.random.normal(0, 1, (32, 2, 200)).astype("float32")
        client_1.infer(background,request_id=i)

        # Wait for the Queue to return the result
        time.sleep(1e-1)
        breakpoint()
        result_1, idx, _ = client_1.get()
        results_2.append(result_1)
        
    # time.sleep(2) 
    # print(results_1)
    # for i, result in enumerate(results_1): 
        
    #     client_2.infer(result,request_id=i)

    #     # Wait for the Queue to return the result
    #     time.sleep(1e-1) 
    #     result_2, idx, _ = client_2.get()


    #     results_2.append(result_2)
        
    results_2 = np.stack(results_2)

    return results_2


