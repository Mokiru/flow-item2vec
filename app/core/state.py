import threading

class AppState:
    def __init__(self):
        self.is_training = False
        self.status_message = "服务就绪，等待训练指令"
        self.lock = threading.Lock()

        self._is_initialized = False
        self._init_error = None

        self.datasource = None
        self.vector_store = None
        self.trainer = None

    def initialize(self) -> bool:
        """
        仅使用时加载
        """
        if self._is_initialized:
            return True

        # 加锁，防止并发触发时重复初始化
        with self.lock:
            if self._is_initialized:
                return True

            try:
                self.status_message = "正在连接基础设施..."
                print(self.status_message)

                from app.infra.data_source import ApiDataSource
                from app.infra.vector_source import MilvusVectorSource
                from app.services.trainer_service import Item2VecTrainer
                from app.core import config

                self.datasource = ApiDataSource(
                    base_url=config.BASE_URL,
                    cookies=config.COOKIES
                )
                self.vector_store = MilvusVectorSource()
                self.trainer = Item2VecTrainer(
                    vector_store=self.vector_store,
                    datasource=self.datasource,
                    dim=config.VECTOR_DIM,
                    lr=config.LEARNING_RATE
                )

                self._is_initialized = True
                self._init_error = None
                self.status_message = "基础设施连接成功，服务就绪。"
                print(self.status_message)
                return True

            except Exception as e:
                self._init_error = str(e)
                self.status_message = f"服务初始化失败: {str(e)}"
                print(f"[严重] {self.status_message}")
                self.datasource = None
                self.vector_store = None
                self.trainer = None
                return False


# 全局单例
app_state = AppState()