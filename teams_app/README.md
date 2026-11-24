# Chữ cái VSL trong Microsoft Teams (bản demo chạy cục bộ)

Ứng dụng Teams có hai chỗ dùng:

- **Tab cá nhân** trên thanh bên trái của Teams;
- **Khung bên cạnh cuộc họp** (meeting side panel), cùng tab trong chat và chi tiết cuộc họp.

Cả hai mở webcam ngay trong Teams và gửi từng khung hình (đã lật gương) tới server Python chạy
trên máy bạn (`https://localhost:53000`). Server dùng
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

Đã cài bản cũ thì tải lại gói mới (version 2.1.0). Teams sẽ cập nhật cùng app nhờ ID không đổi.

## Dùng trong cuộc họp

1. Chạy `python teams_app/server.py` trước khi vào họp.
2. Trong cửa sổ cuộc họp: **+ Apps** (hoặc **More → Apps**) → chọn **VSL Recognition** → trang
   cấu hình báo "Server sẵn sàng" → **Save**. Khung **Ký hiệu VSL** mở bên phải video.
3. Bấm **Bật camera** trong khung, ký từng chữ, giữ ~0,8 giây để gõ.
4. Bấm **Sao chép** rồi dán vào chat cuộc họp.

Không thấy nút **+ Apps**? Thử với một cuộc họp được lên lịch trong lịch Teams. Tôi chưa kiểm tra
cuộc họp tạo nhanh ("Meet now") có cho thêm app hay không.

Lưu ý khi dùng trong cuộc họp:

- **Camera dùng chung:** Teams đang dùng webcam cho cuộc họp.
  - Nếu khung báo **"camera đang bận"**: tắt camera trong cuộc họp, hoặc chọn camera khác trong
    ô chọn camera (chỉ hiện khi máy có từ 2 camera).
  - Việc khung và cuộc họp cùng dùng một camera **chưa được kiểm chứng** trên Teams. Tùy máy và
    driver mà có thể chạy được hoặc không.
- **Chỉ người mở khung thấy chữ.** Người khác cần cài app và chạy server trên máy họ, hoặc bạn
  host server ở địa chỉ HTTPS công khai.
- Chữ **không tự gửi vào chat**. Gửi tự động cần đăng ký app trên Microsoft Entra và có quyền
  quản trị; phần này chưa làm.

> Nếu không thấy mục *Upload a custom app*: tổ chức đang chặn sideload. Admin Teams cần bật
> *Upload custom apps* trong App setup policy, hoặc dùng một tenant Microsoft 365 Developer.

## Tính năng

- Video webcam (lật gương), khung bàn tay và 21 điểm landmark.
- Giao diện tự thu gọn khi khung hẹp (≤ 520 px, như khung bên cạnh cuộc họp): ẩn top-3 và chú
  thích, chữ to vừa khung.
- Chọn camera khi máy có nhiều camera; thông báo rõ khi camera bận hoặc chưa được cấp quyền.
- Nút **Sao chép** văn bản đã gõ.
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
| `static/index.html` | Giao diện tab và khung cuộc họp (webcam, kết quả, Teams JS SDK) |
| `static/config.html` | Trang cấu hình khi thêm app vào cuộc họp (kiểm tra server, lưu tab) |
| `make_cert.py` | Tạo chứng chỉ HTTPS localhost |
| `build_package.py` | Tạo `manifest.json`, icon và gói zip cho Teams |
