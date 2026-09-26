# QSD Compositional Prompt

Nhánh: `qsd-compositional-prompt`, phát triển từ `qsd-prompt`.

## Vì sao chọn hướng này

Kết quả Split-CIFAR100, seed 10961 hiện tại chưa chứng minh lợi ích của QSD hay
phase encoding. QSD đạt 84.08%, thấp hơn L2P 84.48% khoảng 0.40 điểm phần trăm;
forgetting tăng từ 6.3000 lên 6.3556. Circuit không phase encoding giữ Acc gần
baseline (84.46%) và giảm forgetting xuống 5.8111. Đây mới là một seed, không đủ
để kết luận về khác biệt có ý nghĩa thống kê.

Thay đổi này tập trung vào cơ chế học prompt: router quyết định trực tiếp nội
dung prompt thông qua tổ hợp mềm, và prompt cũ được đóng băng. Đây là giả thuyết
cần thực nghiệm, chưa phải kết quả cải thiện Acc/Forgetting đã được đo.

## Cơ chế đã cài đặt

1. ViT đóng băng cung cấp CLS và patch features. Phép chiếu trực giao cố định đưa
   chúng về không gian 32 chiều; trộn density CLS với trung bình density patch.
   Phép chiếu là buffer, không bị hàm khởi tạo Linear của ViT ghi đè.
2. Mỗi nhiệm vụ thêm 5 component prompt, mỗi component gồm 5 token. Sau 10 nhiệm
   vụ có 50 component. Năm đầu định tuyến tạo năm tổ hợp, mỗi tổ hợp 5 token;
   ViT luôn nhận **25 prompt token**. Patch embedding không bị chỉnh sửa.
3. Mỗi đầu dùng các density prompt rank 4 và phép đo pretty-good measurement
   (PGM). Với M component, `A_j = (sigma_j + eps I)/M`, `S = sum_j A_j`,
   `E_j = S^(-1/2) A_j S^(-1/2)`. Tổng `E_j` bằng I; ridge được phân bổ vào
   từng outcome, thay vì chỉ cộng vào S. Gradient inverse square root sử dụng
   đạo hàm phổ có giới hạn hữu hạn khi các trị riêng trùng nhau.
4. `p = 0.5 * Born(rho, E) + 0.5 * softmax(cosine/tau)`; từng đầu dùng
   `sum_j p_j * prompt_j`. Loss phân loại đi trực tiếp qua các trọng số mềm.
   Đây là hỗn hợp xác suất cố định, không phải residual strength học từ zero.
   Có thể thử pure Born bằng `--comp_quantum_mix 1`.
5. Khi sang nhiệm vụ mới, cả prompt và density factors của các bank cũ bị khóa
   bằng `requires_grad=False`, xóa gradient cũ và tạo lại optimizer/scheduler.
   Chỉ bank hiện tại cùng classifier được học. Suy luận định tuyến theo từng
   ảnh trên toàn bộ bank đã thấy, không dùng task ID thật hay majority của batch.
6. Cuối nhiệm vụ, lấy tối đa 64 ứng viên/lớp từ **train loader**; farthest-point
   initialization và 8 vòng k-means trên density tạo tối đa 4 prototype/lớp.
   Chỉ lưu density, query 32 chiều và phân phối định tuyến của router cuối nhiệm
   vụ. Không lưu ảnh hoặc embedding ViT đầy đủ. Đây là bộ nhớ từ dữ liệu, không
   phải phương pháp hoàn toàn không có bộ nhớ.
7. Ở nhiệm vụ sau, lấy 32 prototype cũ/bước để giữ phân phối bằng KL và giữ prompt
   tổ hợp bằng MSE chuẩn hóa. Phân phối đích có xác suất zero tại component mới;
   KL được tính trên toàn bộ pool đang hoạt động, nên có phạt việc chuyển xác
   suất sang component mới. Đích của lớp cũ không bị ghi đè qua các nhiệm vụ.

Loss = CE - 0.1 * weighted cosine pull + 1.0 * retention KL
       + 1.0 * normalized prompt MSE.

Density/POVM được mô phỏng bằng PyTorch thực; không có mạch hay phần cứng lượng
tử. Thiết kế compositional prompt chịu ảnh hưởng của
[CODA-Prompt](https://arxiv.org/abs/2211.13218), kết hợp định tuyến PGM; không phải
bản tái hiện CODA-Prompt và không khẳng định ưu thế lượng tử.

## Chạy trên Kaggle / Linux

Từ checkout chứa mã của nhánh này, cài dependencies hiện có trong repository
(timm 0.6.7). Chạy trực tiếp một process, một GPU:

```bash
# Phương án tăng ngân sách: 7 epoch/task.
DATA_PATH=/kaggle/working/datasets bash train_qsd_compositional_cifar100.sh

# Đối chiếu cùng 5 epoch/task với các kết quả ban đầu.
EPOCHS=5 bash train_qsd_compositional_cifar100.sh

# Đối chứng bỏ Born/POVM, cùng pool, tham số, memory, loss và số epoch.
ROUTER=cosine_comp EPOCHS=7 bash train_qsd_compositional_cifar100.sh

# Đánh giá checkpoint: giữ nguyên cấu hình của lần train.
bash train_qsd_compositional_cifar100.sh --eval
```

Đặt DATA_PATH theo vị trí CIFAR100 đang dùng. Script có output riêng theo router,
epoch và seed. Đặt OUTPUT_DIR riêng khi thay đổi hyperparameter khác để tránh
ghi đè kết quả. `--size` là tham số pool của L2P cũ, không điều khiển capacity của
mode mới; dùng `--comp_components_per_task`.

7 thay vì 5 epoch tăng 40% số bước training, cộng thêm chi phí router và thu
prototype. Nếu lấy thời gian baseline 3h45 làm mốc, riêng tăng epoch tương ứng
khoảng 1h30; đây chỉ là ước lượng tuyến tính, chưa đo trên GPU Kaggle.

CLI cũ `--prompt_router cosine` và `--prompt_router qsd` vẫn giữ thuật toán cũ.
Mode mới hiện chỉ hỗ trợ Split-CIFAR100, class-incremental, một process; không
dùng `torchrun`, task mask, shared prompt hay batchwise routing.

## Đọc kết quả

`results_summary.json` ghi Final Acc, incremental Acc, forgetting, BWT,
old/new accuracy, config và diagnostics **sau training** trên evaluation.
Diagnostics train epoch cuối được lưu riêng, không trộn với final evaluation.

| Chỉ số | Ý nghĩa |
| --- | --- |
| CompKL | KL trên prototype cũ; cần quan sát cùng Acc nhiệm vụ mới |
| CompMSE | Sai lệch prompt tổ hợp trên prototype cũ, chia cho năng lượng prompt đích |
| OldNewMass | Khối xác suất component nhiệm vụ hiện tại nhận trên prototype cũ |
| RoutePeak / RouteEnt | Độ tập trung của tổ hợp; entropy dùng log tự nhiên |
| QSDEnt / QSDPur | Entropy Born / purity của density ảnh |
| QSDStr | Hệ số Born cố định: 0.5 ở preset, 0 ở cosine control |
| POVMError | Sai số max-absolute của sum(E)-I trước chuẩn hóa xác suất |
| Components / MemCount | Số component / prototype đã lưu |

Preset ViT-B có 224,000 tham số prompt/router được cấp phát, trong đó mỗi
nhiệm vụ học một bank 22,400 tham số; classifier học theo L2P. Bộ nhớ prototype
cấp phát khoảng 2 MiB cho 100 lớp. Pool lớn hơn và bộ nhớ prototype là các yếu
tố thay đổi so với baseline; muốn quy cải thiện cho Born/POVM phải so với
`cosine_comp` cùng ngân sách, không chỉ so với L2P 5 epoch.

Ưu tiên kiểm tra cả Acc cuối và forgetting. Forgetting thấp có thể đi kèm mức
học nhiệm vụ mới thấp; vì vậy cần xem thêm incremental Acc, old/new Acc và
đường chéo ma trận accuracy. Tối thiểu lặp lại các seed giống nhau cho các
phương pháp trước khi kết luận. Đóng băng prompt không đảm bảo logits cũ bất
biến: router toàn pool và classifier vẫn có thể làm thay đổi dự đoán.

## Kiểm thử và giới hạn hiện tại

```bash
python -m unittest discover -s tests -v
```

Đã kiểm tra trên CPU PyTorch 2.11, timm 0.6.7: tính đầy đủ/PSD của POVM,
gradcheck cả khi trị riêng trùng nhau, CE truyền gradient tới pure Born router,
prompt cũ không đổi khi optimizer có momentum/weight decay, tính bất biến của
target cũ, retention optimization, đối chứng cosine, checkpoint round-trip và
vòng engine hai nhiệm vụ với ViT nhỏ cho cả hai mode. Kiểm tra đường suy luận
không phụ thuộc task ID và patch tokens không bị thay đổi bởi module prompt.

Chưa chạy training Split-CIFAR100 đầy đủ hoặc benchmark CUDA trong môi trường
hiện tại. Checkpoint evaluation được hỗ trợ; tự động tiếp tục training từ
checkpoint giữa nhiệm vụ chưa được thêm. Các loss hệ số 1.0 và mix 0.5 là
điểm khởi đầu thực nghiệm, chưa được tuning bằng dữ liệu kết quả.
