import numpy as np
import torch

# print(np.arange(10))


# from pymilvus import MilvusClient, DataType
#
# client = MilvusClient(
#     uri="http://localhost:19530",
#     token="hoshino:hoshino"
# )
#
# schema = MilvusClient.create_schema(
#     auto_id=False,
#     enable_dynamic_field=True,
# )


# client.drop_collection(collection_name='test_collection_v1')

# query_vector = client.query(collection_name='test_collection_v1', ids=['6917541900932'], output_fields=['barcode', 'embedding', 'last_train_day'])[0]['embedding']

# res = client.search(
#     collection_name="test_collection_v1",
#     anns_field="embedding",
#     ids=['26920930349131'],
#     limit=3,
#     search_params={"metric_type": "COSINE"}
# )
#
# for hits in res:
#     for hit in hits:
#         print(hit)
