# Chữ cái VSL trong Microsoft Teams (bản demo chạy cục bộ)

Ứng dụng Teams dạng **personal tab**: mở webcam ngay trong Teams, gửi từng khung hình (đã lật
gương) tới server Python chạy trên máy bạn (`https://localhost:53000`). Server dùng
`signbridge.letters`, giống `scripts/live_letter_demo.py`: MediaPipe lấy 21 điểm landmark của bàn
tay, MLP nhỏ phân loại, rồi trả về top-3 chữ cái.

```
Teams (tab, webcam) ──JPEG──▶ server.py (FastAPI, localhost:53000) ──▶ MediaPipe + MLP
        ◀──────── chữ cái, top-3, landmarks ────────
```

## Chuẩn bị (một lần)

Chạy các lệnh trong thư mục gốc của dự án:

```powershell
pip install fastapi uvicorn
python scripts/fetch_mediapipe_models.py
python -X utf8 scripts/record_letters.py --signer s01
python -X utf8 scripts/train_letter_classifier.py
python teams_app/make_cert.py
certutil -user -addstore Root "teams_app\certs\localhost.crt"
python teams_app/build_package.py
```

Trong đó:

- `record_letters.py` thu dữ liệu, `train_letter_classifier.py` train mô hình (ra
  `../models/letters/vsl_letters.pt`). Chưa có mô hình thì trang vẫn mở được nhưng chưa nhận
  dạng.
- `make_cert.py` tạo chứng chỉ HTTPS tự ký cho `localhost` (Teams bắt buộc tab chạy trên HTTPS).
- `certutil` cho Windows tin chứng chỉ đó (chỉ với tài khoản của bạn). Gỡ bỏ:
  `certutil -user -delstore Root localhost`.
- `build_package.py` tạo gói `teams_app/build/vsl-teams-app.zip`.

## Chạy

```powershell
python teams_app/server.py
```

Kiểm tra nhanh trong trình duyệt: mở <https://localhost:53000/>.

## Cài vào Teams

1. **Khởi động lại Teams** sau khi tin chứng chỉ.
2. Teams → **Apps** → **Manage your apps** → **Upload an app** → **Upload a custom app** → chọn
   `teams_app/build/vsl-teams-app.zip`.
3. Nhấn **Add**, mở tab **Nhận dạng VSL**, bấm **Bật camera** và cho phép quyền camera.

> Nếu không thấy mục *Upload a custom app*: tổ chức đang chặn sideload. Admin Teams cần bật
> *Upload custom apps* trong App setup policy, hoặc dùng một tenant Microsoft 365 Developer.

## Tính năng

- Video webcam (lật gương), khung bàn tay và 21 điểm landmark.
- Chữ cái dự đoán, độ tin cậy, top-3.
- Bỏ phiếu theo độ tin cậy trên 6 khung hình, ngưỡng chỉnh được (mặc định 0.60).
- Gõ văn bản: giữ một ký hiệu ~0,8 giây để thêm chữ. "Mũ" sau A/E/O thành Â/Ê/Ô, "Râu" sau O/U
  thành Ơ/Ư. Hạ tay xuống để gõ lặp lại cùng chữ.
- Tự theo giao diện sáng / tối / tương phản cao của Teams.

## Ghi chú

- Chỉ máy đang chạy `server.py` dùng được (tab trỏ tới `localhost`). Muốn người khác cùng dùng:
  host server ở địa chỉ HTTPS công khai (vd. dev tunnel, Azure App Service), rồi build lại gói với
  `python teams_app/build_package.py --url https://<địa-chỉ>`.
- Khung hình chỉ xử lý trong bộ nhớ, không lưu xuống đĩa.
- Độ chính xác phụ thuộc vào dữ liệu bạn tự thu. Xem `outputs/letters/train_report.json` sau khi
  train. Nếu chỉ một người thu, con số đó **không** đo được độ chính xác với người khác.

## Tệp

| Tệp | Vai trò |
|---|---|
| `server.py` | FastAPI: phục vụ trang tab + `POST /api/predict` |
| `static/index.html` | Giao diện tab (webcam, kết quả, Teams JS SDK) |
| `make_cert.py` | Tạo chứng chỉ HTTPS localhost |
| `build_package.py` | Tạo `manifest.json`, icon và gói zip cho Teams |
