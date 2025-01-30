models = ['white_noise_burst', 'gaussian'] 

wildcard_constraints:
    deploymodels = '|'.join(models)

CLI = {
    'white_noise_burst': 'white_noise_burst',
    'gaussian': 'deploy/deploy/cli_infer.py'
}

rule export: 
    input:
        config = 'deploy/deploy/config/export.yaml'
    params:
        cli = lambda wildcards: CLI[wildcards.deploymodels]
    output:
        artefact = directory('output/export/{deploymodels}')
    shell:
        'set -x; cd deploy; poetry run python ../deploy/deploy/cli_export.py \
        --config ../{input.config} --project {params.cli}'

rule export_all:
    input: expand(rules.export.output, deploymodels='white_noise_burst')