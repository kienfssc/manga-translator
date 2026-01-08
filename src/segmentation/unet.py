import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForImageClassification
from torchvision import models


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, up=True) -> None:
        super().__init__()
        if up:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )
        else:
            self.upsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels + skip_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = SCSEModule(ch=out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 2: Conv 3x3, Rate = 2
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 3: Conv 3x3, Rate = 4
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 4: Conv 3x3, Rate = 6
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 5: Global Average Pooling (Global View)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Final Merge
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)

        feat4 = self.avg_pool(x)
        feat4 = self.b4(feat4)
        feat4 = F.interpolate(feat4, size=size, mode="bilinear", align_corners=True)

        # Concat
        x = torch.cat([feat0, feat1, feat2, feat3, feat4], dim=1)
        x = self.project(x)
        return x


class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        #  Channel SE
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // re, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // re, ch, 1),
            nn.Sigmoid(),
        )
        #  Spatial SE
        self.sSE = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Unet_B0(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(Unet_B0, self).__init__()

        backbone = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
        self.encoder = backbone.segformer.encoder

        self.encoder.patch_embeddings[0].proj.stride = (2, 2)
        self.encoder.patch_embeddings[1].proj.stride = (2, 2)
        self.encoder.patch_embeddings[2].proj.stride = (2, 2)
        self.encoder.patch_embeddings[3].proj.stride = (2, 2)

        self.aspp = ASPP(in_channels=256, out_channels=256)

        self.de_layer1 = DecoderBlock(
            in_channels=256, skip_channels=160, out_channels=160
        )

        self.de_layer2 = DecoderBlock(
            in_channels=160, skip_channels=64, out_channels=64
        )

        self.de_layer3 = DecoderBlock(in_channels=64, skip_channels=32, out_channels=32)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

        self.head1 = nn.Conv2d(160, num_classes, kernel_size=1)
        self.head2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.head3 = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # hidden_states: [patch_embed, stage1, stage2, stage3, stage4]
        # s[0] = stage 1 (32 channels)
        # s[1] = stage 2 (64 channels)
        # s[2] = stage 3 (160 channels)
        # s[3] = stage 4 (256 channels)
        s = self.encoder(x, output_hidden_states=True).hidden_states

        bottleneck = self.aspp(s[3])
        d0 = self.de_layer1(bottleneck, s[2])
        d1 = self.de_layer2(d0, s[1])
        d2 = self.de_layer3(d1, s[0])

        main_out = self.final_up(d2)

        if self.training:
            out1 = F.interpolate(
                self.head1(d0), size=(256, 256), mode="bilinear", align_corners=True
            )
            out2 = F.interpolate(
                self.head2(d1), size=(256, 256), mode="bilinear", align_corners=True
            )
            out3 = F.interpolate(
                self.head3(d2), size=(256, 256), mode="bilinear", align_corners=True
            )
            return main_out, out3, out2, out1
        else:
            return main_out


class Unet_B1(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(Unet_B1, self).__init__()
        backbone = SegformerForImageClassification.from_pretrained("nvidia/mit-b1")
        self.encoder = backbone.segformer.encoder

        self.encoder.patch_embeddings[0].proj.stride = (2, 2)
        self.encoder.patch_embeddings[1].proj.stride = (2, 2)

        self.encoder.patch_embeddings[2].proj.stride = (2, 2)
        self.encoder.patch_embeddings[3].proj.stride = (2, 2)

        self.de_layer1 = DecoderBlock(
            in_channels=512, skip_channels=320, out_channels=320
        )
        self.de_layer2 = DecoderBlock(
            in_channels=320, skip_channels=128, out_channels=128
        )
        self.de_layer3 = DecoderBlock(
            in_channels=128, skip_channels=64, out_channels=64
        )
        self.aspp = ASPP(in_channels=512, out_channels=512)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

        self.head1 = nn.Conv2d(320, num_classes, kernel_size=1)
        self.head2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.head3 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        s = self.encoder(x, output_hidden_states=True).hidden_states
        bottleneck = self.aspp(s[3])
        d0 = self.de_layer1(bottleneck, s[2])
        d1 = self.de_layer2(d0, s[1])
        d2 = self.de_layer3(d1, s[0])

        main_out = self.final_up(d2)
        if self.training:
            out1 = F.interpolate(
                self.head1(d0), size=(256, 256), mode="bilinear", align_corners=True
            )
            out2 = F.interpolate(
                self.head2(d1), size=(256, 256), mode="bilinear", align_corners=True
            )
            out3 = F.interpolate(
                self.head3(d2), size=(256, 256), mode="bilinear", align_corners=True
            )
            return main_out, out3, out2, out1
        else:
            return main_out


class Unet_MobileNet_small(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(Unet_MobileNet_small, self).__init__()

        # Load MobileNetV3 Small as the backbone
        # We use .features to get the convolutional layers
        backbone = models.mobilenet_v3_small(weights="DEFAULT").features

        # Encoder Stages (Extracting features at different resolutions)
        # s1: 1/2 size (16 channels)
        self.layer1 = backbone[0:1]
        # s2: 1/4 size (16 channels)
        self.layer2 = backbone[1:2]
        # s3: 1/8 size (24 channels)
        self.layer3 = backbone[2:4]
        # s4: 1/16 size (48 channels)
        self.layer4 = backbone[4:9]
        # s5: 1/32 size (576 channels) - Bottleneck
        self.layer5 = backbone[9:13]

        # ASPP on the bottleneck
        self.aspp = ASPP(in_channels=576, out_channels=256)

        # Decoder Layers
        # de1: 1/32 -> 1/16 (Input: 256, Skip: 48)
        self.de_layer1 = DecoderBlock(
            in_channels=256, skip_channels=48, out_channels=128
        )

        # de2: 1/16 -> 1/8 (Input: 128, Skip: 24)
        self.de_layer2 = DecoderBlock(
            in_channels=128, skip_channels=24, out_channels=64
        )

        # de3: 1/8 -> 1/4 (Input: 64, Skip: 16)
        self.de_layer3 = DecoderBlock(in_channels=64, skip_channels=16, out_channels=32)

        # de4: 1/4 -> 1/2 (Input: 32, Skip: 16)
        self.de_layer4 = DecoderBlock(in_channels=32, skip_channels=16, out_channels=16)

        # Final upsampling to original size (1/2 -> 1/1)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

        # Auxiliary heads for Deep Supervision (matching your Unet_B0/B1 style)
        self.head1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.head2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.head3 = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.layer1(x)  # 1/2
        s2 = self.layer2(s1)  # 1/4
        s3 = self.layer3(s2)  # 1/8
        s4 = self.layer4(s3)  # 1/16
        s5 = self.layer5(s4)  # 1/32 (Bottleneck)

        # Bridge
        bottleneck = self.aspp(s5)

        # Decoder with skip connections
        d0 = self.de_layer1(bottleneck, s4)  # 1/16
        d1 = self.de_layer2(d0, s3)  # 1/8
        d2 = self.de_layer3(d1, s2)  # 1/4
        d3 = self.de_layer4(d2, s1)  # 1/2

        # Final output
        main_out = self.final_up(d3)  # 1/1

        if self.training:
            # Deep Supervision outputs
            out1 = F.interpolate(
                self.head1(d0), size=x.shape[-2:], mode="bilinear", align_corners=True
            )
            out2 = F.interpolate(
                self.head2(d1), size=x.shape[-2:], mode="bilinear", align_corners=True
            )
            out3 = F.interpolate(
                self.head3(d2), size=x.shape[-2:], mode="bilinear", align_corners=True
            )
            return main_out, out3, out2, out1
        else:
            return main_out


class Unet_MobileNet_Large(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(Unet_MobileNet_small, self).__init__()

        # Load MobileNetV3 Small as the backbone
        # We use .features to get the convolutional layers
        backbone = models.mobilenet_v3_small(weights="DEFAULT").features

        # Encoder Stages (Extracting features at different resolutions)
        # s1: 1/2 size (16 channels)
        self.layer1 = backbone[0:1]
        # s2: 1/4 size (16 channels)
        self.layer2 = backbone[1:2]
        # s3: 1/8 size (24 channels)
        self.layer3 = backbone[2:4]
        # s4: 1/16 size (48 channels)
        self.layer4 = backbone[4:9]
        # s5: 1/32 size (576 channels) - Bottleneck
        self.layer5 = backbone[9:13]

        # ASPP on the bottleneck
        self.aspp = ASPP(in_channels=576, out_channels=256)

        # Decoder Layers
        # de1: 1/32 -> 1/16 (Input: 256, Skip: 48)
        self.de_layer1 = DecoderBlock(
            in_channels=256, skip_channels=48, out_channels=128
        )

        # de2: 1/16 -> 1/8 (Input: 128, Skip: 24)
        self.de_layer2 = DecoderBlock(
            in_channels=128, skip_channels=24, out_channels=64
        )

        # de3: 1/8 -> 1/4 (Input: 64, Skip: 16)
        self.de_layer3 = DecoderBlock(in_channels=64, skip_channels=16, out_channels=32)

        # de4: 1/4 -> 1/2 (Input: 32, Skip: 16)
        self.de_layer4 = DecoderBlock(in_channels=32, skip_channels=16, out_channels=16)

        # Final upsampling to original size (1/2 -> 1/1)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

        # Auxiliary heads for Deep Supervision (matching your Unet_B0/B1 style)
        self.head1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.head2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.head3 = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.layer1(x)  # 1/2
        s2 = self.layer2(s1)  # 1/4
        s3 = self.layer3(s2)  # 1/8
        s4 = self.layer4(s3)  # 1/16
        s5 = self.layer5(s4)  # 1/32 (Bottleneck)

        # Bridge
        bottleneck = self.aspp(s5)

        # Decoder with skip connections
        d0 = self.de_layer1(bottleneck, s4)  # 1/16
        d1 = self.de_layer2(d0, s3)  # 1/8
        d2 = self.de_layer3(d1, s2)  # 1/4
        d3 = self.de_layer4(d2, s1)  # 1/2

        # Final output
        main_out = self.final_up(d3)  # 1/1

        if self.training:
            # Deep Supervision outputs
            out1 = F.interpolate(
                self.head1(d0), size=x.shape[-2:], mode="bilinear", align_corners=True
            )
            out2 = F.interpolate(
                self.head2(d1), size=x.shape[-2:], mode="bilinear", align_corners=True
            )
            out3 = F.interpolate(
                self.head3(d2), size=x.shape[-2:], mode="bilinear", align_corners=True
            )
            return main_out, out3, out2, out1
        else:
            return main_out


class Unet_MobileNetLarge(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(Unet_MobileNetLarge, self).__init__()

        # Load MobileNetV3 Large backbone
        backbone = models.mobilenet_v3_large(weights="DEFAULT").features

        # --- Encoder Mapping ---
        # Layer 0-1: Output 1/2 size, 16 channels
        self.layer1 = backbone[0:2]
        # Layer 2-3: Output 1/4 size, 24 channels
        self.layer2 = backbone[2:4]
        # Layer 4-6: Output 1/8 size, 40 channels
        self.layer3 = backbone[4:7]
        # Layer 7-12: Output 1/16 size, 80 channels
        self.layer4 = backbone[7:13]
        # Layer 13-16: Output 1/32 size, 960 channels (Bottleneck)
        self.layer5 = backbone[13:17]

        # ASPP on the high-level features (960 channels)
        self.aspp = ASPP(in_channels=960, out_channels=512)

        # --- Decoder Layers ---
        # de1: 1/32 -> 1/16 (Input: 512, Skip: 80)
        self.de_layer1 = DecoderBlock(
            in_channels=512, skip_channels=112, out_channels=256
        )

        # de2: 1/16 -> 1/8 (Input: 256, Skip: 40)
        self.de_layer2 = DecoderBlock(
            in_channels=256, skip_channels=40, out_channels=128
        )

        # de3: 1/8 -> 1/4 (Input: 128, Skip: 24)
        self.de_layer3 = DecoderBlock(
            in_channels=128, skip_channels=24, out_channels=64
        )

        # de4: 1/4 -> 1/2 (Input: 64, Skip: 16)
        self.de_layer4 = DecoderBlock(in_channels=64, skip_channels=16, out_channels=32)

        # Final upsampling: 1/2 -> 1/1
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

        # Auxiliary heads for training (Deep Supervision)
        self.head1 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.head2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.head3 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.layer1(x)  # 1/2 (16 ch)
        s2 = self.layer2(s1)  # 1/4 (24 ch)
        s3 = self.layer3(s2)  # 1/8 (40 ch)
        s4 = self.layer4(s3)  # 1/16 (80 ch)
        s5 = self.layer5(s4)  # 1/32 (960 ch)
        # Bridge
        bottleneck = self.aspp(s5)

        # Decoder
        d0 = self.de_layer1(bottleneck, s4)  # 1/16

        d1 = self.de_layer2(d0, s3)  # 1/8

        d2 = self.de_layer3(d1, s2)  # 1/4
        d3 = self.de_layer4(d2, s1)  # 1/2

        main_out = self.final_up(d3)  # 1/1

        if self.training:
            # Deep Supervision outputs
            out1 = F.interpolate(
                self.head1(d0), size=x.shape[-2:], mode="bilinear", align_corners=True
            )
            out2 = F.interpolate(
                self.head2(d1), size=x.shape[-2:], mode="bilinear", align_corners=True
            )
            out3 = F.interpolate(
                self.head3(d2), size=x.shape[-2:], mode="bilinear", align_corners=True
            )
            return main_out, out3, out2, out1
        else:
            return main_out


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet_MobileNetLarge(in_channels=3, num_classes=2).to(device)
    dummy_input = torch.randn((2, 3, 256, 256)).to(device)

    model.train()
    outputs = model(dummy_input)
    print(f"Training mode output count: {len(outputs)}")
    print(f"Main output shape: {outputs[0].shape}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Eval mode output shape: {output.shape}")
