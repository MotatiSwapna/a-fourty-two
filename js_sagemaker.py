# -*- coding: utf-8 -*-
"""js_sagemaker.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kr6zm_nYrjd_aNwgDSrX82SMmqu5iNv0
"""

pip install boto3

pip install pickle

import boto3
import pickle

pip install sagemaker

from sagemaker import get_execution_role
role=get_execution_role()

