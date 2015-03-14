"""
Wrap Google Predition API into something that looks
kind of like the standard scikit-learn interface to
learning models.

Derived from Google API example code examples found here:

https://github.com/google/google-api-python-client

@author: Jed Ludlow

"""

from __future__ import print_function

import argparse
import pprint
import time

from apiclient import sample_tools
from oauth2client import client

# Time to wait (in seconds) between successive checks of training status.
SLEEP_TIME = 10

# String to dispaly if OAuth fails.
REAUTH = ("The credentials have been revoked or expired. "
          "Please re-instantiate the predictor to re-authorize.")


def print_header(line):
    """
    Format and print header block sized to length of line

    """
    header_str = '='
    header_line = header_str * len(line)
    print('\n' + header_line)
    print(line)
    print(header_line)


class GooglePredictor(object):
    """
    Prediction engine from the Google Prediction API wrapped
    loosely in the style of sckit-learn.

    """
    def __init__(self, project_id, object_name, model_id, client_secrets):
        # Take advantage of the Google API example tools for
        # credential management which make use of command line
        # argument parsing.
        argparser = argparse.ArgumentParser(add_help=False)
        argparser.add_argument(
            'object_name',
            help="Full Google Storage path of csv data (ex bucket/object)")
        argparser.add_argument(
            'model_id',
            help="Model Id of your choosing to name trained model")
        argparser.add_argument(
            'project_id',
            help="Project Id as shown in Developer Console")

        service, self.flags = sample_tools.init(
            ['GooglePredictor', object_name, model_id, project_id],
            'prediction', 'v1.6', __doc__, client_secrets,
            parents=[argparser],
            scope=(
                'https://www.googleapis.com/auth/prediction',
                'https://www.googleapis.com/auth/devstorage.read_only'))

        self.papi = service.trainedmodels()

    def list(self):
        try:
            # List models.
            print_header("Fetching list of first ten models")
            result = self.papi.list(
                maxResults=10,
                project=self.flags.project_id).execute()
            print("List results:")
            pprint.pprint(result)

        except client.AccessTokenRefreshError:
            print(REAUTH)

    def get_params(self):
        try:
            # Describe model.
            print_header("Fetching model description")
            result = self.papi.analyze(
                id=self.flags.model_id,
                project=self.flags.project_id).execute()
            print("Analyze results:")
            pprint.pprint(result)

        except client.AccessTokenRefreshError:
            print(REAUTH)

    def fit(self, model_type='CLASSIFICATION'):
        try:
            # Start training request on a data set.
            print_header("Submitting model training request")
            body = {
                'id': self.flags.model_id,
                'storageDataLocation': self.flags.object_name,
                'modelType': model_type}
            start = self.papi.insert(
                body=body,
                project=self.flags.project_id).execute()
            print("Training results:")
            pprint.pprint(start)

            # Wait for the training to complete.
            print_header("Waiting for training to complete")
            while True:
                status = self.papi.get(
                    id=self.flags.model_id,
                    project=self.flags.project_id).execute()
                state = status['trainingStatus']
                print("Training state: " + state)
                if state == 'DONE':
                    break
                elif state == 'RUNNING':
                    time.sleep(SLEEP_TIME)
                    continue
                else:
                    raise Exception("Training Error: " + state)

                # Job has completed.
                print("Training completed:")
                pprint.pprint(status)
                break

        except client.AccessTokenRefreshError:
            print(REAUTH)

    def predict(self, X):
        """
        Get model predictions for the examples in X.

        X is either a single list containing the entries in the
        input feature vector for a single example or it is a list of lists
        where each sublist is itself an input feature vector.

        """
        try:
            # Make some predictions using the newly trained model.
            # print_header("Making some predictions")
            out = []
            for sample in X:
                body = {'input': {'csvInstance': sample}}
                result = self.papi.predict(
                    body=body,
                    id=self.flags.model_id,
                    project=self.flags.project_id).execute()
                if 'outputLabel' in result:
                    out.append(result['outputLabel'])
                elif 'outputValue' in result:
                    out.append(float(result['outputValue']))
            return out

        except client.AccessTokenRefreshError:
            print(REAUTH)

    def delete(self):
        try:
            # Delete model.
            print_header('Deleting model')
            result = self.papi.delete(
                id=self.flags.model_id,
                project=self.flags.project_id).execute()
            print("Model deleted.")
            return result

        except client.AccessTokenRefreshError:
            print(REAUTH)
