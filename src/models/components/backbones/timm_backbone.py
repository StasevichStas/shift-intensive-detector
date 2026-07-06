import timm
from torch import nn
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

class TimmFPNBackbone(nn.Module):
    def __init__(self,
                 model: str = 'mobilenetv3_large_100',
                 pretrained: bool = True,
                 fpn_out_channels: int = 256):
        super().__init__()

        # 1. Создаем временную модель, чтобы узнать структуру ее фичей
        temp_model = timm.create_model(model, pretrained=False, features_only=True)
        num_stages = len(temp_model.feature_info.channels())
        
        # 2. Берем последние 4 этапа (подходит и для 4-этапных, и для 5-этапных моделей)
        out_indices = tuple(range(max(0, num_stages - 4), num_stages))

        # 3. Создаем финальный бэкбон
        self.backbone = timm.create_model(
            model,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        in_channels_list = self.backbone.feature_info.channels()

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=fpn_out_channels,
        )

        self.out_channels = fpn_out_channels

    def forward(self, x):
        features = self.backbone(x)
        features = OrderedDict(
            (str(i), feat)
            for i, feat in enumerate(features)
        )
        return self.fpn(features)