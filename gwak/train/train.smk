# Define a dictionary with the required parameters for each model
model_params = {
    'sine_gaussian': {
        'signal': 'SineGaussian',
        'model': 'LSTM_AE_SPLIT',
        'num_timesteps': 10,
        'bottleneck': 5
    },
    # 'background': {
    #     'signal': None,
    #     'model': 'FAT',
    #     'num_timesteps': 20,
    #     'bottleneck': 10
    # }
}

import os
from pathlib import Path
# from gwak.train.cli import train

# rule train:
#     params:
#         bottleneck = lambda wildcards: model_params[wildcards.data]['bottleneck'],
#         model = lambda wildcards: model_params[wildcards.data]['model'],
#         data = lambda wildcards: model_params[wildcards.data]['signal'],
#     output:
#         model_file = 'output/{data}/model.pt',
#         artefact = directory('output/{data}/')
#     run:
#         os.makedirs(output.artefact, exist_ok=True)
#         train(
#             data_type=params.data,
#             model_name=params.model,
#             model_file=output.model_file,
#             artefacts=Path(output.artefact)
#             )

rule train:
    params:
        bottleneck = lambda wildcards: model_params[wildcards.data]['bottleneck'],
        model = lambda wildcards: model_params[wildcards.data]['model'],
        data = lambda wildcards: model_params[wildcards.data]['signal'],
    output:
        artefact = directory('output/{data}/')
    shell:
        'python train/cli.py fit \
            --trainer.strategy ddp_find_unused_parameters_true \
            --trainer.max_epochs 1 \
            --trainer.logger WandbLogger \
            --trainer.logger.save_dir {output.artefact}'

rule train_all:
    input:
        expand(rules.train.output, data=model_params.keys())