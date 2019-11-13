from argparse import ArgumentParser
from datetime import datetime
from io import StringIO  
import logging
import json
import os
import pandas as pd
import requests
import sys

import boto3
import botocore
from botocore.errorfactory import ClientError

from fasttext_model import TextPreprocessor, BookMovieClassifier

logging.basicConfig(level="INFO")

BUCKET='nytimeshainingr'

stg_id = os.environ['AWS_KEY_ID'] 
stg_key = os.environ['AWS_SECRET_KEY'] 
TIMES_API_SECRET = os.environ['TIMES_API_SECRET'] 


def write_df_s3(df, bucket, key):
    try: 
        csv_buffer = StringIO()
        df.to_csv(csv_buffer)
        s3_resource = boto3.resource('s3', aws_access_key_id=stg_id, aws_secret_access_key=stg_key)
        s3_resource.Object(bucket, key).put(Body=csv_buffer.getvalue())
    except ClientError:
        logging.info('could not write')

def download_file_s3(bucket, key, rename):
    try: 
        s3 = boto3.resource("s3", aws_access_key_id=stg_id, aws_secret_access_key=stg_key)
        s3.Bucket(bucket).download_file(key, rename)
    except ClientError:
        logging.info('could not download')

def get_fiction_list(date):
    api_link = "https://api.nytimes.com/svc/books/v3/lists/%s/hardcover-fiction.json?api-key=%s" %(date, TIMES_API_SECRET) 
    data = requests.get(api_link).json() 
    if data.get("results") is None:
        logging.info("no data found, call was not successful")
    else:
        return pd.read_json(json.dumps(data['results']['books']))

def valid_date(s):
    try: 
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except ValueError:
        msg="Not a valid date: '{0}'".format(s)
        raise argparse.ArgumentTypeError(msg)

def main(args):
    parser = ArgumentParser()
    parser.add_argument("-date", required=True, type=valid_date, help="Date of BestSeller- format YYYY-MM-DD")
    parser.add_argument("-download-nyt", type=str, nargs='?', help="Whether to read the data in from NYT API at given date")
    parser.add_argument("-predict", type=str, nargs='?', help="Whether to predict on data on given date")
    parser.add_argument("-model-config", type=str, nargs='?', help="Path to model config for predicting")

    args = parser.parse_args()
    DATE = args.date
    today_df = get_fiction_list(DATE)
    if 'download_nyt' in args:
        write_df_s3(today_df, "nytimeshainingr", DATE)
    
    if 'predict' in args:
        assert('model_config' in args)
        with open(args.model_config,'r') as f:
            modelConfig = json.load(f)

        textPreprocessor = TextPreprocessor()
        model = BookMovieClassifier(textPreprocessor)
        logging.info("Loading model")
        model.load_model(modelConfig['weightsPath'])

        download_file_s3(BUCKET, DATE, DATE) 
        prediction_df = pd.read_csv(DATE, encoding='utf-8')
        prediction_df['movie_prediction'] = model.predict(prediction_df['description'])

        write_df_s3(prediction_df, BUCKET, DATE+'_predictions')

"""
#### USAGE EXAMPLE ###
source cred/cred.sh
python pipeline.py -date 2019-08-05 -download-nyt -predict -model-config config/model_deploy_config.json


"""
if __name__=='__main__':
    main(sys.argv[1:])
