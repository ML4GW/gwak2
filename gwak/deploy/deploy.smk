deployclasses = ['export', 'infer']  # List of deployment types

wildcard_constraints:
    deploytype = '|'.join(deployclasses)

CLI = {
    'export': 'deploy/deploy/cli_export.py',
    'infer': 'deploy/deploy/cli_infer.py'
}

rule deploy: 
    input:
        config = 'deploy/deploy/config/{deploytype}.yaml'
    params:
        cli = lambda wildcards: CLI[wildcards.deploytype]
    output:
        artefact = directory('/home/hongyin.chen/anti-gravity/gwak/gwak/output/{deploytype}')
    shell:
        'cd deploy; poetry run python ../{params.cli} --config ../{input.config}'

rule deploy_all:
    input: expand(rules.deploy.output, deploytype='export')