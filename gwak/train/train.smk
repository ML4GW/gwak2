signalclasses = ['bbh', 'sine_gaussian_lf', 'sine_gaussian_hf', 'sine_gaussian']
backgroundclasses = ['background', 'glitches']
dataclasses = signalclasses+backgroundclasses

wildcard_constraints:
    dataclass = '|'.join([x for x in dataclasses])


rule train:
    input:
        config = 'train/configs/{datatype}.yaml'
    log:
        artefact = directory('output/{datatype}/')
    shell:
        'python train/cli.py fit --config {input.config} \
            --trainer.logger.save_dir {log.artefact}'

rule train_all:
    input:
        expand(rules.train.log, datatype='background')