import logging

from lightning.pytorch.cli import LightningCLI

from ml4gw import waveforms


def sum_args(a, b):
    return float(a) + float(b)
def product_args(a, b):
    return int(a) * int(b)
def cudaify(devices):
    # TODO: model support for training on across multiple devices
    # needs a change here
    return f"cuda:{devices[0]}"

class GwakSignalCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):

        parser.link_arguments(
            'data.init_args.sample_rate',
            'data.init_args.waveform.init_args.sample_rate',
            apply_on='parse'
        )
        parser.link_arguments(
            'trainer.devices',
            'model.init_args.device_str',
            compute_fn=cudaify,
            apply_on='parse'
        )
        parser.link_arguments(
            ('data.init_args.sample_rate', 'data.init_args.kernel_length'),
            'model.init_args.num_timesteps',
            compute_fn=product_args,
            apply_on='parse'
        )
        parser.link_arguments(
            ('data.init_args.kernel_length', 'data.init_args.fduration'),
            'data.init_args.waveform.init_args.duration',
            compute_fn=sum_args,
            apply_on='parse'
        )
        


def cli_main(args=None):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Started')

    cli = GwakSignalCLI(
            save_config_kwargs={'overwrite': True},
            args=args
    )


if __name__ == '__main__':
    cli_main()