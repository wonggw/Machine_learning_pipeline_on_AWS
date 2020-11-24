import io,json,boto3
from sagemaker.amazon.common import write_numpy_to_dense_tensor
import numpy as np

arr = np.array([ 0.,2.,14.,16.,12.,6.,0.,0.,0.,1.,10.,8.,14.,
        16.,1.,0.,0.,0.,0.,0.,10.,15.,2.,0.,0.,0.,
         0.,2.,16.,12.,0.,0.,0.,0.,0.,3.,16.,12.,0.,
         0.,0.,0.,0.,0.,11.,16.,2.,0.,0.,0.,7.,10.,
        15.,15.,2.,0.,0.,3.,13.,11.,7.,2.,0.,0.]).reshape(1,64)
true_val = 3
buf = io.BytesIO()
write_numpy_to_dense_tensor(buf,arr)
buf.seek(0)

client=boto3.client('sagemaker-runtime', region_name='us-west-2')

response = client.invoke_endpoint(
    EndpointName='Demo-Endpoint',
    ContentType='application/x-recordio-protobuf',
    Accept='application/json',
    Body=buf)

print(json.loads(response['Body'].read().decode('UTF-8')))