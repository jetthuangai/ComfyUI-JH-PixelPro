# Setup guide — ComfyUI-JH-PixelPro

Guide dành cho **JH** và các agent (Codex / Claude Code) khi lần đầu đưa pack vào trạng thái dev-ready.

> Nếu bạn chỉ **cài pack** (consumer), đọc README.md thôi là đủ. File này cho **maintainer**.

## 1. Khởi tạo git repo riêng cho pack

Yêu cầu của User: pack này push GitHub **độc lập**, không liên quan tới repo ComfyUI bao ngoài.

> ⚠️ **Lưu ý sandbox:** Khi Cowork agent scaffold pack trong sandbox, `git init` bị lỗi giữa chừng do sandbox bảo vệ git metadata. Nếu bạn thấy thư mục `.git/` trống/lỗi hiện có, hãy xoá nó trước khi init mới.

Thực hiện từ terminal thật (PowerShell / bash trên máy JH):

```bash
cd ComfyUI/custom_nodes/ComfyUI-JH-PixelPro

# Nếu có .git/ lỗi sẵn — xoá đi
rm -rf .git          # macOS/Linux
# hoặc
rmdir /s /q .git     # Windows PowerShell

# Init lần đầu
git init -b main
git add .
git commit -m "chore: scaffold ComfyUI-JH-PixelPro (Phase 1)"

# Trỏ tới remote GitHub (JH tự tạo repo trống trước)
git remote add origin git@github.com:<user>/ComfyUI-JH-PixelPro.git
git push -u origin main
```

## 2. Verify pack không dính vào git của ComfyUI

```bash
cd ComfyUI
git check-ignore -v custom_nodes/ComfyUI-JH-PixelPro
# Kỳ vọng: .gitignore:8:/custom_nodes/	custom_nodes/ComfyUI-JH-PixelPro
```

Nếu kết quả rỗng → pack sẽ bị ComfyUI git track. Sửa `.gitignore` của ComfyUI ngay.

## 3. Cài dev dependencies

```bash
cd ComfyUI/custom_nodes/ComfyUI-JH-PixelPro
pip install -r requirements-dev.txt
```

## 4. Smoke test sau khi cài

```bash
# Test pack load được trong ComfyUI
# (khởi động ComfyUI, mở web UI, kiểm tra menu image/pixelpro/ có xuất hiện)

# Pytest — lúc này chưa có test nào pass vì chưa có core math,
# nhưng lệnh phải chạy không lỗi import.
pytest --collect-only
```

## 5. Git boundary checklist (mỗi lần commit)

- [ ] `git rev-parse --show-toplevel` chạy trong pack → phải ra đường dẫn `ComfyUI-JH-PixelPro/`, **không phải** `ComfyUI/`.
- [ ] `git status` không liệt kê file ngoài pack.
- [ ] `git remote -v` chỉ trỏ tới repo của pack, không phải upstream ComfyUI.

## 6. Xem tiếp

- `.agent-hub/10_plan/master-plan.md` — toàn cảnh 9 node + 3 phase.
- `.agent-hub/20_specs/S-01-frequency-separation.md` — spec đầu tiên đang làm.
- `.agent-hub/30_tasks/` — task list chi tiết.
- `.agent-hub/00_charter/folder-layout.md` — ràng buộc umbrella folder.
