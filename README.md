# Pytorch_YOLOv5_seg_Deepsort
 YOLOv5 and Deepsort

1. Detection tracker

![demo](demo_yolov5_deepsort.gif)

2. Detection，segmentation tracker

![demo](demo_yolov5_seg_deepsort.gif)

## Download
### Deepsort
Download [ckpt.t7](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6) and copy to deep_sort_pytorch/deep_sort/deep/checkpoint/

### YOLOv5
Downlod Yolov5 model and copy to yolov5/weights/

Detection weights

|Model |
| ------ |
|[YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt) |
|[YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt) |
|[YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt) |
|[YOLOv5l](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt) |
|[YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt)  |
|[YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n6.pt)|
|[YOLOv5s6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s6.pt) |
|[YOLOv5m6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m6.pt)|
|[YOLOv5l6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt) |
|[YOLOv5x6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x6.pt) |

Segmentation weights

|Model |
| ------ |
|[YOLOv5n-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-seg.pt)|
|[YOLOv5s-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-seg.pt)|
|[YOLOv5m-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m-seg.pt) |
|[YOLOv5l-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-seg.pt) |
|[YOLOv5x-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x-seg.pt) |

## Tracking sources

Tracking can be run on most video formats

```bash
# detection tracker
python3 track.py --source ... --show-vid  # show live inference results as well

# segmentation tracker
python3 track_seg.py --source ... --show-vid  # show live inference results as well
```

- Video:  `--source file.mp4`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`
