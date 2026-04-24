import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import requests


class DataSource(ABC):
    """
    数据源抽象接口（纯同步版本）
    设计原则：不保存任何状态，所有的上下文（游标、时间）均由调用方通过参数传入。
    """

    @abstractmethod
    def fetch_batch(
            self,
            batch_size: int,
            sid: str,
            trade_id: int,
            date: str
    ) -> Tuple[bool, str, int, List[List[str]]]:
        """
        同步拉取一批原始订单数据。

        返回: Tuple[是否成功拿到有效数据, 新的sid, 新的trade_id, 订单列表]
        注意：即使返回 False (远端没数据了)，也必须原样返回传入的 sid 和 trade_id。
        """
        pass

    @abstractmethod
    def get_sampling_data(
            self,
            barcodes: List[str],
            date: str
    ) -> Dict[str, Dict[str, np.float64]]:
        """
        同步获取指定 barcodes 的采样数据（负采样概率等）。
        """
        pass


@dataclass
class OrderData:
    sid: str = ""
    tradeId: int = 0
    barcodeList: List[List[str]] = field(default_factory=list)

    def __post_init__(self):
        data = []
        for ele in self.barcodeList:
            data.append(json.loads(ele))
        self.barcodeList = data


@dataclass
class OrderResponse:
    success: str
    code: int
    data: OrderData


@dataclass
class BarcodePopularity:
    barcode: str
    finalWeight: float
    samplingProb: float
    lnProbCorrection: float

    def __post_init__(self):
        # 确保所有数值类型都是numpy float64
        self.finalWeight = np.float64(self.finalWeight)
        self.samplingProb = np.float64(self.samplingProb)
        self.lnProbCorrection = np.float64(self.lnProbCorrection)


@dataclass
class PopularityResponse:
    success: str
    code: int
    data: List[BarcodePopularity]


class ApiDataSource(DataSource):
    """
    同步数据获取
    """

    def __init__(self, base_url: str = "", cookies: dict = None):
        self.base_url = base_url
        self.order_endpoint = "/algo/train/order"
        self.popularity_endpoint = "/algo/barcode/popularity"
        self.cookies = cookies or {}

    def fetch_batch(
            self,
            batch_size: int,
            sid: str,
            trade_id: int,
            date: str
    ) -> Tuple[bool, str, int, List[List[str]]]:

        raw_orders, new_sid, new_trade_id, is_success = self._request_orders(sid, trade_id, date)

        if not is_success or not raw_orders:
            # 没拿到数据，原样返回调用方传进来的游标
            return False, sid, trade_id, []

        # 清洗数据
        clean_orders = self._normalize_orders(raw_orders)
        if not clean_orders:
            # 接口返回了但全是脏数据，游标已更新，但视作本批无效
            return False, new_sid, new_trade_id, []

        # 成功拿到有效数据
        return True, new_sid, new_trade_id, clean_orders

    def get_sampling_data(
            self,
            barcodes: List[str],
            date: str
    ) -> Dict[str, Dict[str, np.float64]]:

        return self._request_popularity(barcodes, date)

    def _request_orders(self, sid: str, trade_id: int, date: str) -> Tuple[List, str, int, bool]:
        """
        接口调用
        :param sid: 卖家账号
        :param trade_id: 订单id
        :param date: 日期
        :return: 订单数据
        """
        params = {"sid": sid, "tradeId": trade_id, "orderDate": date}
        url = f"{self.base_url}{self.order_endpoint}"
        try:
            response = requests.post(url, json=params, cookies=self.cookies, timeout=15)
            if response.status_code != 200:
                print(f"[DataSource] 订单API失败: {response.status_code}")
                return [], sid, trade_id, False

            data = response.json()
            order_response = OrderResponse(
                success=data['success'],
                code=data['code'],
                data=OrderData(**data['data'])
            )

            if order_response.data and order_response.data.barcodeList:
                return (
                    order_response.data.barcodeList,
                    order_response.data.sid,
                    order_response.data.tradeId,
                    True
                )
            return [], sid, trade_id, False

        except requests.exceptions.Timeout:
            print("[DataSource] 订单API超时")
            return [], sid, trade_id, False
        except Exception as e:
            print(f"[DataSource] 订单API异常: {e}")
            return [], sid, trade_id, False

    def _request_popularity(self, barcodes: List[str], date: str) -> Dict[str, Dict[str, np.float64]]:
        """真实的 HTTP 请求逻辑"""
        if not barcodes:
            return {}

        url = f"{self.base_url}{self.popularity_endpoint}"
        try:
            response = requests.post(
                url,
                json={"barcodes": barcodes, "updateTime": date},
                cookies=self.cookies,
                timeout=10
            )
            if response.status_code != 200:
                print(f"[DataSource] 采样API失败: {response.status_code}")
                return self._get_default_sampling(barcodes)

            data = response.json()
            popularity_response = PopularityResponse(
                success=data['success'],
                code=data['code'],
                data=[BarcodePopularity(**item) for item in data['data']]
            )

            result = {}
            for item in popularity_response.data:
                if item.barcode in barcodes:
                    result[item.barcode] = {
                        'samplingProb': item.samplingProb,
                        'finalWeight': item.finalWeight,
                        'lnProbCorrection': item.lnProbCorrection
                    }

            # 补齐缺失的 barcode
            for bc in barcodes:
                if bc not in result:
                    result[bc] = {
                        'samplingProb': np.float64(1e-8),
                        'finalWeight': np.float64(-18.42),
                        'lnProbCorrection': np.float64(np.log(1e-8))
                    }
            return result

        except Exception as e:
            print(f"[DataSource] 采样API异常: {e}")
            return self._get_default_sampling(barcodes)

    def _get_default_sampling(self, barcodes: List[str]) -> Dict[str, Dict[str, np.float64]]:
        """无数据情况"""
        default = {
            'samplingProb': np.float64(1e-8),
            'finalWeight': np.float64(-18.42),
            'lnProbCorrection': np.float64(np.log(1e-8))
        }
        return {bc: default.copy() for bc in barcodes}

    def _normalize_orders(self, orders) -> List[List[str]]:
        """数据清洗"""
        normalized = []
        for order in orders:
            valid = [bc for bc in order if bc and isinstance(bc, str)]
            if valid:
                normalized.append(valid)
        return normalized
