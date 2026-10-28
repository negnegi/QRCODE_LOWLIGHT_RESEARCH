# WeChat QR model files

These four files are the pre-trained models of the WeChat QR code reader
(`cv2.wechat_qrcode_WeChatQRCode`), used as the deep-learning reference method.
They are **not** part of this work. They were obtained from:

https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode

| File | Role |
|---|---|
| `detect.prototxt`, `detect.caffemodel` | CNN-based QR detector |
| `sr.prototxt`, `sr.caffemodel` | Super-resolution model |

See the original repository for their license.
