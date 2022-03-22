import datetime

import pandas as pd
import requests
from google.cloud import storage
from pandas import DataFrame


def get_vm_custom_envs(meta_key: str):
    response = requests.get(
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/{}".format(meta_key),
        headers={'Metadata-Flavor': 'Google'},
    )

    data = response.text

    return data


def write_data(df: DataFrame, sink_name: str):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(get_vm_custom_envs(sink_name))

    csv_name = "xgb-pred.csv" if sink_name == 'PREDICTIONS_SINK' else "{}-xgb-pred.csv".format(
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))

    bucket.blob(csv_name).upload_from_string(
        df.to_csv(header=False, index=False), "text/csv")


def save_to_db(df: DataFrame):
    write_data(df, 'PREDICTIONS_SINK')
    write_data(df, 'PREDICTIONS_OVER_TIME_SINK')


def read_storage_csv(file_name: str):
    return pd.read_csv(
        'gs://{}/{}'.format(get_vm_custom_envs("PREP_SINK"), file_name)
    )


def upper_limits():
    print(' ')
    print('______________________________________________________________')


def under_limits():
    print('______________________________________________________________')
    print(' ')
