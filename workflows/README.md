# Workflows

Sample workflow để demo từng node của pack `ComfyUI-JH-PixelPro`. Mỗi file `.json` là 1 graph load thẳng vào ComfyUI (menu **Load** → chọn file) hoặc drag-drop vào canvas.

## Danh sách

| File | Node chính | Mô tả |
|---|---|---|
| [S-01-frequency-separation.json](S-01-frequency-separation.json) | `JHPixelProFrequencySeparation` | Split ảnh thành low-freq (smooth) + high-freq (detail). Demo retouch chuyên nghiệp. Có Note node in-canvas giải thích invariant `low + high = original` và lý do `high` pin có giá trị âm. |

## Usage

ComfyUI `LoadImage` node chỉ scan folder `ComfyUI/input/`. Để chạy sample workflow:

1. Copy `workflows/sample_portrait.jpg` → `ComfyUI/input/sample_portrait.jpg`.
2. Menu **Load** trong ComfyUI → chọn `S-01-frequency-separation.json` (hoặc drag-drop JSON vào canvas).
3. Nhấn **Queue Prompt** — 2 PreviewImage sẽ hiện `low` (smooth) và `high` (detail, dark vì giá trị âm).

## Image credits

- `sample_portrait.jpg` — Photo by [cottonbro studio](https://www.pexels.com/@cottonbro/) from Pexels. Source: <https://www.pexels.com/photo/close-up-photo-of-woman-s-beautiful-face-6567969/>. License: [Pexels Content License](https://www.pexels.com/license/) (free for commercial + non-commercial redistribution, attribution not required but ghi nhận best practice).
