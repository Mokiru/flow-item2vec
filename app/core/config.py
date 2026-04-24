# Milvus 配置
MILVUS_URI = "http://172.17.0.3:19530"
COLLECTION_NAME = "barcode_bundle_collection_v1"

# 模型配置
VECTOR_DIM = 128  # 向量维度
LEARNING_RATE = 0.01
MAX_ID_LENGTH = 511  # barcode长度

# 训练配置
BATCH_SIZE = 2000

# API 配置
BASE_URL = 'http://localhost:8080'
COOKIES = {

}

MAX_ITEMS_PER_ORDER = 50
MICRO_BATCH_SIZE = 20000