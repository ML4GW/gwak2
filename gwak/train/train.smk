rule train:
    input:
        config = 'train/configs/{data_type}.yaml'
    output:
        artefact = directory('output/{data_type}/')
    shell:
        'python train/cli.py fit --config {input.config} \
            --trainer.logger.save_dir {output.artefact}'

rule train_all:
    input:
        expand(rules.train.output, data_type='sine_gaussian')