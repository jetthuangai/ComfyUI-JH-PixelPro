# Workflows

Sample workflow để demo từng node của pack `ComfyUI-JH-PixelPro`. Mỗi file `.json` là 1 graph load thẳng vào ComfyUI (menu **Load** → chọn file) hoặc drag-drop vào canvas.

## Danh sách

| File | Node chính | Mô tả |
|---|---|---|
| [S-01-frequency-separation.json](S-01-frequency-separation.json) | `JHPixelProFrequencySeparation` | Split ảnh thành low-freq (smooth) + high-freq (detail). Demo retouch chuyên nghiệp. Có Note node in-canvas giải thích invariant `low + high = original` và lý do `high` pin có giá trị âm. |

## Image credits

- `sample_portrait.jpg` — TBD (sẽ fill trong T-04b artifact task).
