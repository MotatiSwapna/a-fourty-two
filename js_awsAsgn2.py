#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import boto3
import pandas as pd
import os
s3=boto3.resource('s3')
my_bucket='mysagemakerex2'
bucket=s3.Bucket(my_bucket)
csvfiles=[]
#listing all files present in s3 bucket
totalFiles=list(i for i in bucket.objects.all())
csvfiles=[]
c=0
#counting number of csv fies  present in s3 bucket
for i in totalFiles:
    if i.key.find(".csv")==1:
        c+=1
print("total number of csv files in bucket is ",c)
#extrexting csv files
for i in range(1,c+1):
    csvfiles.append(s3.Object(my_bucket, ".csv".format(i)))
file1 = pd.read_csv(csvfiles[0].get()["Body"])
file2 = pd.read_csv(csvfiles[1].get()["Body"])
file3 = pd.read_csv(csvfiles[2].get()["Body"])
file4 = pd.read_csv(csvfiles[3].get()["Body"])


# In[2]:


file1.head()


# In[3]:


file2.head()


# In[4]:


file3.head()


# In[5]:


file4.head()


# In[6]:


entireData=pd.concat([file1,file2,file3,file4])
print(entireData.head())
print(entireData.tail())


# In[7]:


entireData.to_csv("EntireData.csv",index="False")
bucket.upload_file( "EntireData.csv" , "EntireData.csv")


# In[8]:


import sagemaker
from sagemaker.tuner import IntegerParameter
from sagemaker.tuner import CategoricalParameter
from sagemaker.tuner import ( ContinuousParameter, HyperparameterTuner,)
region = boto3.Session().region_name
smclient = boto3.Session().client("sagemaker")

role = sagemaker.get_execution_role()
prefix = 'sagemaker'


# In[9]:


#we have total dataset already ,if not we can get it from s3 bucket using the following
entireData = pd.read_csv(s3.Object(my_bucket, "EntireData.csv").get()["Body"], index_col=False)


# In[10]:


import numpy as np


# In[11]:


train_data, validation_data, test_data = np.split(entireData.sample(frac=1, random_state=777),[int(0.6 * len(entireData)), int(0.8 * len(entireData))])


# In[12]:


train_data.to_csv("train.csv", index=False, header=False)
validation_data.to_csv("validation.csv", index=False, header=False)
test_data.to_csv("test.csv", index=False, header=False)


# In[13]:


boto3.Session().resource("s3").Bucket(my_bucket).Object(os.path.join(prefix, "train/train.csv")).upload_file("train.csv")
boto3.Session().resource("s3").Bucket(my_bucket).Object(os.path.join(prefix, "validation/validation.csv")).upload_file("validation.csv")


# In[ ]:





# In[14]:


prefix = 'train.csv'
input_data_config = [
    {
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": "s3://{}/{}".format(my_bucket, prefix),
            }
        },
        "TargetAttributeName": "price_range",
    }
]

output_data_config = {"S3OutputPath": "s3://{}/{}/output".format(my_bucket, prefix)}


# In[15]:


from sagemaker.inputs import TrainingInput
s3_input_train = TrainingInput(
    s3_data="s3://{}/{}/train".format(my_bucket, prefix), content_type="csv"
)
s3_input_validation = TrainingInput(
    s3_data="s3://{}/{}/validation".format(my_bucket, prefix), content_type="csv"
)


# In[16]:


from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.image_uris import retrieve
from time import strftime,gmtime
sess = sagemaker.Session()

container = retrieve("xgboost", region, "latest")

xgb = sagemaker.estimator.Estimator(
    container,
    role,
    base_job_name="xgboost-random-search",
    instance_count=1,
    instance_type="ml.m4.xlarge",
    output_path="s3://{}/{}/output".format(my_bucket, prefix),
    sagemaker_session=sess,
)

xgb.set_hyperparameters(
    eval_metric="auc",
    objective="binary:logistic",
    num_round=10,
    rate_drop=0.3,
    tweedie_variance_power=1.4,
)
objective_metric_name = "validation:auc"


# In[17]:


hyperparameter_ranges = {
    "alpha": ContinuousParameter(0.01, 10, scaling_type="Logarithmic"),
    "lambda": ContinuousParameter(0.01, 10, scaling_type="Logarithmic"),
}


# In[ ]:


tuner_log = HyperparameterTuner(
    xgb,
    objective_metric_name,
    hyperparameter_ranges,
    max_jobs=3,
    max_parallel_jobs=3,
    strategy="Random",
)

tuner_log.fit(
    {"train": s3_input_train, "validation": s3_input_validation},
    include_cls_metadata=False,
    job_name="xgb-randsearch-" + strftime("%Y%m%d-%H-%M-%S", gmtime()),
)


# In[ ]:


boto3.client("sagemaker").describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuner_log.latest_tuning_job.job_name
)["HyperParameterTuningJobStatus"]


# In[ ]:


hyperparameter_ranges_linear = {
    "alpha": ContinuousParameter(0.01, 10, scaling_type="Linear"),
    "lambda": ContinuousParameter(0.01, 10, scaling_type="Linear"),
}


# In[ ]:


tuner_linear = HyperparameterTuner(
    xgb,
    objective_metric_name,
    hyperparameter_ranges_linear,
    max_jobs=5,
    max_parallel_jobs=5,
    strategy="Random",
)

tuner_linear.fit(
    {"train": s3_input_train, "validation": s3_input_validation},
    include_cls_metadata=False,
    job_name="xgb-linsearch-" + strftime("%Y%m%d-%H-%M-%S", gmtime()),
)


# In[ ]:


boto3.client("sagemaker").describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuner_linear.latest_tuning_job.job_name
)["HyperParameterTuningJobStatus"]

