from abc import ABC, abstractmethod

# from pymilvus import MilvusClient, DataType, MilvusException



class VectorSource(ABC):

    @abstractmethod
    def get_vectors(self, barcodes):
        """
        基于barcode列表获取向量
        """
        pass

    @abstractmethod
    def upsert_new_vectors(self, entities):
        """
        保存结果
        """
        pass


class MilvusVectorSource(VectorSource):

    def __init__(self):
        self.client = MilvusClient(uri=config.MILVUS_URI, user='hoshino', password='hoshino')
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        if self.client.has_collection(config.COLLECTION_NAME):
            print(f"集合 {config.COLLECTION_NAME} 已存在，跳过创建。")
            return

        schema = self.client.create_schema()

        # 主键
        schema.add_field(
            field_name="barcode",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=config.MAX_ID_LENGTH
        )

        # 商品向量
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=config.VECTOR_DIM
        )

        schema.add_field(
            field_name="last_train_day",
            datatype=DataType.VARCHAR,
            max_length=16
        )

        index_params = self.client.prepare_index_params()

        # 向量索引
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 256, "ef": 128}
        )

        index_params.add_index(field_name="last_train_day", index_type="AUTOINDEX")
        index_params.add_index(field_name="barcode", index_type="AUTOINDEX")

        self.client.create_collection(
            collection_name=config.COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

    def _ensure_loaded(self):
        """
        每次真正操作前调用。如果已加载，瞬间返回；如果未加载，自动加载。
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.client.load_collection(collection_name=config.COLLECTION_NAME)
                return True
            except MilvusException as e:
                print(f"[Milvus] 确保 Load 失败 (第 {attempt + 1} 次重试): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # 等 2 秒再试，给 Milvus 恢复的时间
                else:
                    print("[Milvus] 多次重试 Load 仍然失败，放弃本次操作。")
                    return False
        return False

    def get_vectors(self, barcodes):
        if not barcodes:
            return {}

        if not self._ensure_loaded():
            return {}

        res = self.client.query(
            collection_name=config.COLLECTION_NAME,
            ids=barcodes,
            output_fields=["barcode", "embedding", "last_train_day"]
        )

        data_map = {}
        for item in res:
            data_map[item['barcode']] = {
                'w': item['embedding'],
                'last_train_day': item['last_train_day']
            }
        return data_map

    def upsert_new_vectors(self, entities):
        if not entities:
            return

        if not self._ensure_loaded():
            print("[Milvus] 写入拦截：集合不可用，丢弃本批数据！")
            return

        clean_data = []
        for barcode, emb, today in entities:
            clean_data.append({
                'barcode': barcode,
                'embedding': np.array(emb, dtype=np.float32),
                'last_train_day': today
            })

        self.client.upsert(
            collection_name=config.COLLECTION_NAME,
            data=clean_data
        )
        print(f"[Milvus] 更新成功 {len(clean_data)} 个商品")
