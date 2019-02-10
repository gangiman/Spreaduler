from spreaduler import ParamsSheet
from train_attention import train
from options import get_parser


class YourParamsSheet(ParamsSheet):
    """
    Your model Params Sheet class
    """
    params_sheet_id = '...'

    client_credentials = {
        "type": "service_account",
        "project_id": "....",
        "private_key_id": "....",
        "private_key": """-----BEGIN PRIVATE KEY-----
........
-----END PRIVATE KEY-----""",
        "client_email": "yourworker@yourproject.iam.gserviceaccount.com",
        "client_id": "....",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://accounts.google.com/o/oauth2/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/"
                                "yourworker%40yourproject.iam.gserviceaccount.com"
    }

    def __init__(self, parser):
        writable_metrics_and_types = {
             'your model precision': float
        }
        super(YourParamsSheet, self).__init__(
            parser,
            writable_column_types=writable_metrics_and_types,
            experiment_id_column='exp_hash')


if __name__ == '__main__':
    params = YourParamsSheet(get_parser())
    params.exec_loop(train)
