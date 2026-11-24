# SignBridge — Ngôn ngữ ký hiệu tiếng Việt (VSL)

Dự án nghiên cứu cá nhân hướng tới **ngôn ngữ ký hiệu liên tục → văn bản tiếng Việt → giọng nói**.
Repo gồm:

1. **Nhận dạng bảng chữ cái VSL** (`signbridge/letters/`): MediaPipe lấy 21 điểm landmark của
   bàn tay → đặc trưng hình dạng bàn tay → MLP nhỏ → giữ tay để gõ chữ. Có bản chạy trong
   Microsoft Teams (`teams_app/`). Mô hình train trên dữ liệu bạn tự thu (chỉ lưu landmark,
   không lưu ảnh). **Bản đầu tiên:** khoảng 50 % khung hình đúng với người lạ (2 người ký, đo
   offline), chưa đo khi chạy live. Xem [docs/ai_validation_status.md](docs/ai_validation_status.md) §3.
2. **Demo ASL chào hỏi** (`signbridge/signs/`): 29 ký hiệu ASL → câu tiếng Anh / tiếng Việt
   → giọng nói.
3. **Lõi nghiên cứu `signbridge/`** cho dịch ngôn ngữ ký hiệu liên tục (CSLT): pipeline dữ liệu,
   tiền xử lý video, giao diện mô hình độc lập với dataset, metric WER, TTS tiếng Việt.

---

## Cài đặt

Yêu cầu: Python 3.10+, webcam (cho demo). GPU không bắt buộc.

```bash
pip install -r requirements-research.txt
python scripts/fetch_mediapipe_models.py      # model MediaPipe ra ../models/mediapipe (ngoài Git)
```

Mọi file model (`.pt`, `.task`, giọng Piper) nằm ở `../models/`, dataset ở `../datasets/`. Cả hai
đều ngoài Git.

---

## 1. Bảng chữ cái VSL

Bộ nhãn mặc định gồm 23 chữ cái tĩnh (A B C D Đ E G H I K L M N O P Q R S T U V X Y) và 2 dấu:
**Mũ** (A/E/O → Â/Ê/Ô) và **Râu** (O/U → Ơ/Ư). Chữ cần chuyển động và dấu thanh chưa hỗ trợ.

### Chạy nhanh

**1. Thử demo** (cần mô hình ở `../models/letters/vsl_letters.pt`):

```powershell
python -X utf8 scripts/live_letter_demo.py                   # giữ tay yên ~0,8 s để gõ một chữ
python -X utf8 scripts/live_letter_demo.py --threshold 0.5   # nhận dạng yếu thì hạ ngưỡng (mặc định 0.6)
python -X utf8 scripts/live_letter_demo.py --hold 1.0        # gõ nhầm nhiều thì giữ lâu hơn
```

Phím: `SPACE` dấu cách, `BACKSPACE` xóa, `C` xóa hết, `Q` thoát. Hạ tay xuống (hoặc đổi chữ) để gõ
lặp lại cùng một chữ.

**2. Quay thêm rồi train lại** (cách cải thiện độ chính xác tốt nhất):

```powershell
python -X utf8 scripts/record_letters.py --signer s01        # số lượt cộng dồn với lần trước
python -X utf8 scripts/train_letter_classifier.py --train-only-signer ext01
```

`--train-only-signer` dành cho người ký được nhập từ ảnh (`ext01`: mỗi chữ chỉ có 1 bản ghi). Họ
chỉ dùng để học, không dùng để test. Thiếu cờ này trainer sẽ báo lỗi "need at least 2". Nếu
chỉ có dữ liệu tự thu thì bỏ cờ đi.

**3. Đo trên người lạ** (giữ nguyên một người để test):

```powershell
python -X utf8 scripts/train_letter_classifier.py --test-signer ext01 --out ../models/letters/exp.pt --report outputs/letters/exp.json
```

Dùng `--out` / `--report` riêng để không ghi đè mô hình đang dùng.

**4. Nhập ảnh chữ cái có sẵn** (mỗi thư mục là một chữ, chỉ lưu landmark):

```powershell
python -X utf8 scripts/import_letter_images.py <thư_mục_ảnh> --signer ext02 --map Dd=DD Rau=RAU mu_=MU
```

`--map` đổi tên thư mục sang mã nhãn. Thêm `--mirror` nếu ảnh chưa lật gương.

**5. Chạy trong Microsoft Teams / trình duyệt:**

```powershell
python teams_app/server.py      # https://localhost:53000/
```

Lỗi cổng 53000 đang bận thì tắt server cũ trước. Cài đặt Teams xem
[teams_app/README.md](teams_app/README.md).

### Chi tiết

Thu dữ liệu (`record_letters.py`):

- `SPACE` quay một lượt: 1,5 s đếm ngược, rồi 2,5 s ghi.
- `N` / `P` chuyển chữ tiếp / trước, `X` xóa lượt vừa quay, `Q` thoát.
- Hết danh sách thì quay vòng lại từ đầu. Nên có **≥ 5 lượt mỗi chữ**, và đổi nhẹ vị trí, góc,
  khoảng cách giữa các vòng.
- Dữ liệu lưu ở `../datasets/vsl_letters/<signer>/<LABEL_ID>/take_*.npz`.

Train (`train_letter_classifier.py`):

- Giữ lại nguyên lượt quay để làm val/test, ghi báo cáo vào `outputs/letters/train_report.json`.
- Nếu chỉ một người thu, kết quả **không** phản ánh độ chính xác với người khác.
- Thu thêm người và dùng `--test-signer s02` để đo trên người lạ.
- Mỗi lần train ghi đè `../models/letters/vsl_letters.pt`. Muốn giữ bản cũ thì sao lưu trước
  hoặc dùng `--out`.
- Đọc `outputs/letters/train_report.json`: `per_label_frame_top1` (độ chính xác từng chữ),
  `top_confusions` (cặp chữ hay nhầm), `caveat` (giới hạn của phép đo).

---

## 2. Lõi nghiên cứu `signbridge/` (CSLT)

How2Sign (ASL, CC BY-NC 4.0) chỉ được dùng để kiểm chứng hạ tầng; không ánh xạ ASL sang VSL.

```bash
python scripts/fetch_how2sign_subset.py annotations
python scripts/fetch_how2sign_subset.py fetch val_rgb_front_clips --n 24 --seed 0
python scripts/audit_how2sign.py && python scripts/build_manifests.py && python scripts/validate_pipeline.py
python -m pytest              # test realdata tự bỏ qua nếu chưa tải dữ liệu
```

Đọc văn bản tiếng Việt thành giọng nói (offline, Piper):

```bash
python scripts/fetch_tts_voice.py                             # tải giọng vi_VN-vais1000-medium ra ../models (ngoài Git)
python -m signbridge.tts "Xin chào các bạn" -o xin_chao.wav --play
```

### Demo live: ký hiệu ASL chào hỏi → câu tiếng Anh / tiếng Việt → giọng nói

29 ký hiệu ASL (chào hỏi, giới thiệu), ký từng ký hiệu một, hạ tay giữa các ký hiệu. Mô hình
huấn luyện trên ASL Citizen (Microsoft Research License, chỉ nghiên cứu phi thương mại). Câu
được ghép bằng bảng cụm từ viết tay, **không phải** mô hình dịch. Kết quả đo:
[docs/asl_greetings_demo.md](docs/asl_greetings_demo.md).

```bash
python scripts/fetch_asl_citizen.py list && python scripts/fetch_asl_citizen.py fetch --glosses HELLO NAME ...
python scripts/extract_sign_features.py && python scripts/train_sign_classifier.py
python scripts/live_sign_demo.py                 # webcam, đọc tiếng Anh (giọng Windows)
python scripts/live_sign_demo.py --speak vi      # đọc câu tiếng Việt (Piper)
```

Tài liệu: [docs/dataset_audit.md](docs/dataset_audit.md) ·
[docs/phoenix_audit.md](docs/phoenix_audit.md) ·
[docs/asl_greetings_demo.md](docs/asl_greetings_demo.md) ·
[docs/architecture.md](docs/architecture.md) ·
[docs/validation_report.md](docs/validation_report.md) ·
[docs/ai_validation_status.md](docs/ai_validation_status.md) ·
[docs/cslr_data_requirements.md](docs/cslr_data_requirements.md) ·
[docs/slt_data_requirements.md](docs/slt_data_requirements.md) ·
[docs/tts_implementation.md](docs/tts_implementation.md) ·
[docs/next_development_roadmap.md](docs/next_development_roadmap.md)

---

## Cấu trúc thư mục

```
├── signbridge/
│   ├── letters/         # bảng chữ cái VSL: đặc trưng landmark, MLP, nhận dạng, gõ chữ, bản ghi
│   ├── signs/           # demo ASL chào hỏi
│   ├── data/ video/ features/ text/ models/   # lõi CSLT
│   ├── metrics/ tts/    # WER, giọng nói
│   └── live_io.py       # camera (khung mới nhất), vẽ chữ tiếng Việt
├── scripts/             # thu dữ liệu, train, demo, tải dữ liệu, kiểm chứng
├── teams_app/           # tab Microsoft Teams cho bảng chữ cái
├── tests/               # pytest (synthetic + realdata)
├── docs/                # audit dataset, kiến trúc, báo cáo kiểm chứng
├── configs/             # cấu hình tiền xử lý
└── LICENSE, THIRD_PARTY_NOTICES.md, CHANGELOG.md
```

Không nằm trong Git: `outputs/`, `../datasets/` (How2Sign, PHOENIX, ASL Citizen, chữ cái tự thu),
`../models/` (MediaPipe, mô hình chữ cái, mô hình ASL, giọng Piper).

---

## License

Code và tài liệu: [MIT](LICENSE). Thư viện bên thứ ba (Piper là GPL-3.0), model MediaPipe và các
dataset có điều khoản riêng: xem [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Mô hình ASL
train trên ASL Citizen chỉ dùng cho nghiên cứu phi thương mại và không được phân phối. Lịch sử
phiên bản: [CHANGELOG.md](CHANGELOG.md).

## Ghi công dữ liệu

- How2Sign: Duarte et al., *How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign
  Language*, CVPR 2021. CC BY-NC 4.0, chỉ dùng cho nghiên cứu phi thương mại.
- ASL Citizen (Microsoft Research), RWTH-PHOENIX-Weather 2014 T: xem
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
