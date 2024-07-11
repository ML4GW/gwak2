rule train:
    # params:
    #     bottleneck = lambda wildcards: model_params[wildcards.data]['bottleneck'],
    #     model = lambda wildcards: model_params[wildcards.data]['model'],
    #     data = lambda wildcards: model_params[wildcards.data]['signal'],
    input:
        config = 'config.yaml'
    output:
        artefact = directory('output/sine_gaussian/')
    shell:
        'python train/cli.py fit --config {input.config}'

            # --trainer.strategy ddp_find_unused_parameters_true \
            # --trainer.max_epochs 1 \
            # --trainer.logger WandbLogger \
            # --trainer.logger.save_dir {output.artefact}'