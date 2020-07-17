from argparse import ArgumentParser


def config_opts(parser):
    parser.add_argument('-project_dir', '--project_dir', required=True, type=str, default='')
    parser.add_argument('-data', '--data', type=str, required=True)
    parser.add_argument('-config', '--config', required=False, help='Config file path')
    parser.add_argument('-project_log', '--project_log', type=str, default='')
    parser.add_argument('-debug', '--debug', type=bool, default=False)


def privacy_metric_opts(parser):
    parser.add_argument('-nums_server', '--nums_server', type=int, default=5)


def de_identity_opts(parser):
    parser.add_argument('-nums_party', '--nums_party', type=int, default=5)


def get_default_argument(desc='default'):
    parser = ArgumentParser(description=desc)
    config_opts(parser)
    if desc == 'privacy-metric':
        privacy_metric_opts(parser)
    elif desc == 'de-identity':
        de_identity_opts(parser)
    else:
        privacy_metric_opts(parser)
        de_identity_opts(parser)
    args = parser.parse_args()
    config = vars(args)
    return config
