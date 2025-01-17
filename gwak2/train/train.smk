import os

signalclasses = ['bbh', 'sine_gaussian_lf', 'sine_gaussian_hf']
backgroundclasses = ['background', 'glitches']
dataclasses = signalclasses+backgroundclasses

wildcard_constraints:
    dataclass = '|'.join([x for x in dataclasses])

CLI = {
    'background': 'train/cli_base.py',
    'glitches': 'train/cli_base.py',
    'bbh': 'train/cli_base.py',
    'sine_gaussian_lf': 'train/cli_signal.py',
    'sine_gaussian_hf': 'train/cli_signal.py',
    }

rule train:
    input:
        config = 'train/configs/{datatype}.yaml'
    params:
        cli = lambda wildcards: CLI[wildcards.datatype]
    log:
        artefact = directory('output/{datatype}/')
    shell:
        'python {params.cli} fit --config {input.config} \
            --trainer.logger.save_dir {log.artefact}'

rule train_all:
    input:
        expand(rules.train.log, datatype='bbh')
