import psutil
import socket

from pathlib import Path

from hermes.aeriel.serve import serve
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


def infer_process(
    client,
    batcher, 
    loop_verbose: int=100
): 
    
    results = []
    for i, background in enumerate(batcher):

        if i % loop_verbose == 0: 
            logging.info(f"Producing inference on {i}th iteration!")

        client.infer(
            background,
            request_id=i,
        )

        # Wait for the Queue to return the result
        time.sleep(1e-1) 
        result, dx, _ = client.get()
        results.append(result)
        
    results = np.stack(results)

    return results

