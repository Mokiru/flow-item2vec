import os
import json
import threading
import time
import traceback
from datetime import timedelta
from app.core.state import app_state


class TrainingTaskManager:
    CHECKPOINT_FILE = "train_checkpoint.json"
    LOG_INTERVAL = 50

    @staticmethod
    def _load_checkpoint() -> dict:
        if os.path.exists(TrainingTaskManager.CHECKPOINT_FILE):
            try:
                with open(TrainingTaskManager.CHECKPOINT_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"sid": "", "trade_id": 0, "date": ""}

    @staticmethod
    def _save_checkpoint(sid: str, trade_id: int, date: str):
        with open(TrainingTaskManager.CHECKPOINT_FILE, 'w') as f:
            json.dump({"sid": sid, "trade_id": trade_id, "date": date}, f)

    @staticmethod
    def _clear_checkpoint():
        if os.path.exists(TrainingTaskManager.CHECKPOINT_FILE):
            os.remove(TrainingTaskManager.CHECKPOINT_FILE)

    @staticmethod
    def _execute_training_loop(today):
        if not app_state.initialize():
            app_state.is_training = False
            return

        try:
            app_state.is_training = True
            batch_count = 0

            trainer = app_state.trainer
            datasource = app_state.datasource
            train_date = today.strftime("%Y-%m-%d") # 训练日期
            query_date = (today - timedelta(days=1)).strftime("%Y-%m-%d") # 接口查询日期

            ckpt = TrainingTaskManager._load_checkpoint()
            current_sid = ckpt.get("sid", "")
            current_trade_id = ckpt.get("trade_id", 0)
            last_train_date = ckpt.get("date", "")

            if last_train_date and last_train_date != train_date:
                app_state.status_message = f"检测到跨天({last_train_date}->{train_date})，清空历史游标。"
                current_sid = ""
                current_trade_id = 0
                TrainingTaskManager._clear_checkpoint()

            app_state.status_message = "任务启动，同步拉取数据中..."

            # 计时
            t_start = time.perf_counter()
            last_log_time = t_start
            last_log_batch_count = 0

            while True:
                # 拉取订单
                success, new_sid, new_trade_id, orders = datasource.fetch_batch(
                    batch_size=trainer.batch_size,
                    sid=current_sid,
                    trade_id=current_trade_id,
                    date=query_date
                )

                if not success:
                    t_end = time.perf_counter()
                    total_time = t_end - t_start
                    avg_time = total_time / batch_count if batch_count > 0 else 0
                    app_state.status_message = f"数据耗尽。共 {batch_count} 批，总耗 {total_time:.1f}s，均耗 {avg_time:.2f}s/批。"
                    TrainingTaskManager._clear_checkpoint()
                    break

                # 更新内存中的游标
                current_sid = new_sid
                current_trade_id = new_trade_id

                if not orders:
                    continue  # 接口有返回但全是脏数据，跳过本批

                # 拉取采样数据
                unique_barcodes = list(set(bc for order in orders for bc in order))
                sampling_data = datasource.get_sampling_data(unique_barcodes, query_date)

                # 数据=>Trainer
                res = trainer.train_with_data(orders, sampling_data, train_date)

                if res == 0:
                    continue

                batch_count += 1
                if batch_count % TrainingTaskManager.LOG_INTERVAL == 0:
                    now = time.perf_counter()
                    interval_batches = batch_count - last_log_batch_count
                    interval_time = now - last_log_time
                    avg_batch_time = interval_time / interval_batches

                    print(f"[性能监控] 完成 {batch_count} 批 | "
                          f"近 {interval_batches} 批耗时: {interval_time:.2f}s | "
                          f"均耗: {avg_batch_time:.2f}s/批")

                    app_state.status_message = f"训练中 (完成 {batch_count} 批，近 {interval_batches} 批均耗 {avg_batch_time:.2f}s)"

                    last_log_time = now
                    last_log_batch_count = batch_count
                else:
                    app_state.status_message = f"训练中... (已完成 {batch_count} 批)"

                TrainingTaskManager._save_checkpoint(current_sid, current_trade_id, train_date)

        except Exception as e:
            app_state.status_message = f"异常中断: {str(e)}"
            traceback.print_exc()
        finally:
            app_state.is_training = False

    @staticmethod
    def start_training_if_idle(today) -> bool:
        if app_state.is_training:
            return False
        thread = threading.Thread(target=TrainingTaskManager._execute_training_loop, args=(today, ), daemon=True)
        thread.start()
        return True
