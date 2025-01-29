from deploy.export import export

from jsonargparse import ArgumentParser, ActionConfigFile


def build_parser():
    
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(export)
    
    return parser

def main(args=None):
    
    parser = build_parser()
    args = parser.parse_args()
    args = args.as_dict()

    for key, item in args.items():
        print(key, item)
    export(**args)
    
    
if __name__ == "__main__":
    main()