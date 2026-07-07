import torch
import torch.nn as nn
from torchvision.ops import nms

class CustomYoloDecoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 2,           # Игрок / Фон
        num_anchors_per_cell: int = 2,  # Например, один вертикальный, один квадратный
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 20,
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors_per_cell
        self.score_thresh = box_score_thresh
        self.nms_thresh = box_nms_thresh
        self.detections_per_img = box_detections_per_img
        
        # Забираем твой быстрый бэкбон
        self.backbone = backbone
        
        # ⚠️ ВАЖНО: Определи, сколько каналов на выходе у твоего бэкбона на последнем слое.
        # Для mobilenetv3_large_100 это обычно 160 или 960 (в зависимости от настроек фпн).
        # Предположим, что после твоего универсального FPN у тебя фиксированно 256 каналов:
        in_channels = 256 
        
        # Каждому анкору нужно предсказать: 4 координаты (x, y, w, h) + 1 objectness + num_classes
        prediction_elements = 4 + 1 + self.num_classes
        out_channels = self.num_anchors * prediction_elements
        
        # Ультра-легкая YOLO-голова: всего 2 свертки вместо тяжелых слоев RetinaNet!
        self.yolo_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.SiLU(), # Быстрая активация
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1) # Финальный предикт
        )

    def forward(self, images, targets=None):
        # 1. Если пришёл список картинок (list), склеиваем его в 4D-тензор [B, C, H, W]
        if isinstance(images, list):
            images = torch.stack(images, dim=0)
            
        # 2. Теперь images — это чистокровный Tensor, и timm/ghostnet его скушает
        features = self.backbone(images)
        
        if isinstance(features, dict):
            features = list(features.values())[-1]
        elif isinstance(features, list):
            features = features[-1]
            
        prediction = self.yolo_head(features)
        
        if self.training:
            return {"yolo_output": prediction}
            
        return self._postprocess(prediction, images)

    def _postprocess(self, prediction, images):
        # Здесь пишется легкий и прямолинейный парсинг тензора:
        # Разворачиваем prediction в [Batch, Сетка_H * Сетка_W * Анкоры, Элементы]
        # Применяем порог score_thresh и встроенный быстрый torchvision.ops.nms
        
        # Для Санити-чека и быстрого старта возвращаем пустой шаблон или базовый постпроцесс:
        batch_size = prediction.shape[0]
        detections = []
        for _ in range(batch_size):
            detections.append({
                "boxes": torch.zeros((0, 4), device=prediction.device),
                "labels": torch.zeros((0,), dtype=torch.long, device=prediction.device),
                "scores": torch.zeros((0,), device=prediction.device)
            })
        return detections