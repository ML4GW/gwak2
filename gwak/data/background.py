import h5py
import subprocess

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from background_utils import (
    get_conincident_segs,
    get_background,
    create_lcs,
    omicron_bashes,
    glitch_merger
)


def run_bash(bash_file):

    subprocess.run(
        ["bash", f"{bash_file}"], 
    )


def gwak_background(
    ifos, 
    state_flag, 
    channels, 
    frame_type,
    ana_start,
    ana_end, 
    sample_rate, 
    save_dir: Path, 
    # seg_count, # 
    **omi_paras,
):

    segs = get_conincident_segs(
        ifos=ifos,
        start=ana_start,
        stop=ana_end,
        state_flag=state_flag, 
    )


    for seg_num, (seg_start, seg_end) in enumerate(segs):

        strains = get_background(
            seg_start=seg_start, 
            seg_end=seg_end,
            ifos=ifos, 
            channels=channels, 
            frame_type=frame_type, 
            sample_rate=sample_rate,
        )

        seg_dur = seg_end-seg_start
        file_name = f"background-{seg_start}-{seg_dur}" 

        with h5py.File(save_dir / file_name) as g:

            for ifo in ifos:
                g.create_dataset(ifo, data=strains[ifo])


        bash_files = [] # List of omicron commands to excute in background.  
        if omi_paras is not None:
            
            for ifo, frametype in zip(ifos, frame_type):

                create_lcs(
                    ifo=ifo,
                    frametype=f"{ifo}_{frametype}",
                    start_time=seg_start,
                    end_time=seg_end,
                    output_dir=omi_paras.out_dir / f"Segs_{seg_num:05d}", 
                    urltype="file"
                )

            bash_scripts = omicron_bashes(
                ifos= ifos,
                start_time=seg_start,
                end_time=seg_end,
                project_dir= omi_paras.out_dir / f"Segs_{seg_num:05d}",
                # INI
                q_range= omi_paras.q_range,
                frequency_range= omi_paras.frequency_range,
                frame_type= frame_type,
                channels= channels,
                cluster_dt= omi_paras.cluster_dt,
                sample_rate= sample_rate,
                chunk_duration= omi_paras.chunk_duration,
                segment_duration= omi_paras.segment_duration,
                overlap_duration= omi_paras.overlap_duration,
                mismatch_max= omi_paras.mismatch_max,
                snr_threshold= omi_paras.snr_threshold,
            )

            for bash_script in bash_scripts:
                bash_files.append(bash_script)


        with ThreadPoolExecutor(max_workers=8) as e: # 8 workers
        
            for bash_file in bash_files:
                e.submit(run_bash, bash_file)

    # Generate a glitch_info.h5 file that stores omicron informations 
    glitch_merger = glitch_merger(
        ifos=ifos,
        omicron_path=omi_paras.out_dir,
        channels=channels
    )