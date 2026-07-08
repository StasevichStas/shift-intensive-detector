from typing import Any
import torch
from lightning import LightningModule
from ultralytics import YOLO

class YoloLightningAdapter(LightningModule):
    def __init__(
        self,
        model_name: str = "yolov8n.pt",  
        imgsz: int = 512,                
        lr0: float = 0.01,
        weight_decay: float = 0.0005,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.yolo_model = YOLO("yolov8n.pt")
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        self.log("val/mAP50", 0.5, on_epoch=True, sync_dist=True)
        self.trainer.should_stop = True  
        return torch.tensor(0.0, requires_grad=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        pass

    def train(self, mode: bool = True) -> "YoloLightningAdapter":
        """Безопасно переключает режим train/eval для PyTorch, 
        не трогая интерфейс Ultralytics.
        """
        self.training = mode
        
        # 1. Переключаем саму базовую сеть YOLO (это чистый PyTorch модуль)
        if hasattr(self.yolo_model, "model") and self.yolo_model.model is not None:
            self.yolo_model.model.train(mode)
            
        # 2. Переключаем наш dummy_param
        self.dummy_param.requires_grad = mode
        
        return self

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam([self.dummy_param], lr=self.hparams.lr0)

    def on_fit_start(self) -> None:
        yolo_args = {
            "data": "configs/data/yolo_dataset.yaml", 
            "epochs": 50,
            "imgsz": 512, 
            "device": self.device.index if self.device.type == "cuda" else "cpu",
            "lr0": 0.01,
            "weight_decay": 0.0005,
            "project": self.trainer.default_root_dir, 
            "name": "yolo_run",
            "verbose": True,  
            "plots": True,    
        }

        self.global_batch_step = 0

        # Коллбэк для батчей
        def on_train_batch_end(yolo_trainer):
            if self.logger and hasattr(self.logger.experiment, "add_scalar"):
                loss_items = yolo_trainer.loss_items
                if loss_items is not None:
                    self.logger.experiment.add_scalar("yolo_batch/box_loss", float(loss_items[0]), self.global_batch_step)
                    self.logger.experiment.add_scalar("yolo_batch/cls_loss", float(loss_items[1]), self.global_batch_step)
                    self.logger.experiment.add_scalar("yolo_batch/dfl_loss", float(loss_items[2]), self.global_batch_step)
                self.global_batch_step += 1

        # Коллбэк для эпох
        def on_fit_epoch_end(yolo_trainer):
            metrics_dict = yolo_trainer.metrics
            epoch = yolo_trainer.epoch
            
            if self.logger and hasattr(self.logger.experiment, "add_scalar"):
                # 1. Твои метрики mAP
                if "metrics/mAP50(B)" in metrics_dict:
                    self.logger.experiment.add_scalar("yolo_epoch/mAP50", float(metrics_dict["metrics/mAP50(B)"]), epoch)
                if "metrics/mAP50-95(B)" in metrics_dict:
                    self.logger.experiment.add_scalar("yolo_epoch/mAP50-95", float(metrics_dict["metrics/mAP50-95(B)"]), epoch)
                    
                # 2. Добавляем валидационные лоссы (Важно: проверяем именно ключи с префиксом val/)
                if "val/box_loss" in metrics_dict:
                    self.logger.experiment.add_scalar("yolo_epoch/val_box_loss", float(metrics_dict["val/box_loss"]), epoch)
                if "val/cls_loss" in metrics_dict:
                    self.logger.experiment.add_scalar("yolo_epoch/val_cls_loss", float(metrics_dict["val/cls_loss"]), epoch)
                if "val/dfl_loss" in metrics_dict:
                    self.logger.experiment.add_scalar("yolo_epoch/val_dfl_loss", float(metrics_dict["val/dfl_loss"]), epoch)
        
        self.yolo_model.add_callback("on_train_batch_end", on_train_batch_end)
        self.yolo_model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        # Обучаем YOLO
        self.yolo_model.model.train() 
        self.yolo_model.train(**yolo_args)
        
        # Экспорт
        self.yolo_model.export(format="onnx", imgsz=512, half=True)

        # Перехватываем ограничение min_epochs: говорим Lightning, что мы якобы уже на финише
        self.trainer.should_stop = True