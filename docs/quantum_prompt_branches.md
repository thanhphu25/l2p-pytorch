# Hai hướng quantum-inspired trên nền L2P

Tài liệu này mô tả hai nhánh thí nghiệm độc lập được phát triển từ L2P:

1. `qsd-prompt`: thay cơ chế **truy xuất prompt** bằng phân biệt trạng thái lượng tử.
2. `phase-interference-l2p`: giữ nguyên truy xuất L2P và thay đổi **patch đầu vào** bằng giao thoa pha do prompt điều khiển.

Hai nhánh kiểm tra hai giả thuyết khác nhau. Không nên gộp chúng trong thí
nghiệm đầu tiên vì khi accuracy thay đổi sẽ không xác định được đóng góp đến từ
retrieval hay patch transformation.

## 1. Tóm tắt khác biệt

| Thành phần | L2P gốc | `qsd-prompt` | `phase-interference-l2p` |
|---|---|---|---|
| Query | CLS của ViT đóng băng | CLS và patch tokens của ViT đóng băng | CLS của ViT đóng băng như L2P |
| Chọn prompt | Cosine similarity | Cosine + Born probability từ PGM/POVM | Cosine similarity nguyên bản |
| Prompt pool | Không đổi | Không đổi | Không đổi |
| Patch embedding | Không đổi | Không đổi | Điều chế bằng trọng số từ phase interference hoặc control MLP |
| Bộ nhớ qua task | Không có | Một density state và phân phối đo trung bình cho mỗi task | Không có bộ nhớ mới |
| Loss mới | Không có | Measurement-retention KL | Không có loss phụ mới |
| Baseline nội bộ | Cosine L2P | `--prompt_router cosine` | `--patch_transform none` |
| Câu hỏi nghiên cứu | Prompting có giảm quên? | Quantum state discrimination có chọn prompt và giữ phép đo tốt hơn cosine? | Giao thoa pha giữa vùng ảnh có tốt hơn reweighting cổ điển cùng dung lượng? |

## 2. Baseline L2P chung

Với ảnh `x`, L2P thực hiện hai đường forward:

1. ViT đóng băng sinh query `q(x)`.
2. So sánh cosine giữa query và các prompt key.
3. Chọn top-`K` prompt trong prompt pool.
4. Nối các prompt đã chọn với patch embeddings.
5. Chạy ViT có prompt và classifier.

Loss baseline trong repository là:

```text
L_L2P = L_CE - pull_constraint_coeff * reduce_sim
```

Hai nhánh bên dưới chỉ thay đúng một trục của luồng này.

---

## 3. Nhánh `qsd-prompt`: Quantum State Discrimination Prompt Retrieval

### 3.1. Git và phạm vi

```text
Branch: qsd-prompt
Base: main
Commits:
  b609330 Implement quantum state prompt retrieval
  d32668a Add compact continual learning summaries
```

Nhánh này thay retrieval. Nó không dùng phase-interference patch transform.

### 3.2. Giả thuyết

Cosine retrieval biểu diễn mỗi ảnh và prompt bằng một vector duy nhất. Điều này
có thể mất thông tin đa vùng và không mô hình hóa được sự mơ hồ khi nhiều prompt
phù hợp đồng thời.

QSD-Prompt kiểm tra giả thuyết:

> Biểu diễn ảnh và prompt bằng mixed density states, sau đó phân biệt các prompt
> bằng một phép đo PGM/POVM, có thể tạo tín hiệu retrieval giàu hơn cosine; giữ
> ổn định phép đo này qua các task có thể giảm forgetting.

Đây là mô phỏng quantum-inspired trên GPU cổ điển, không phải tuyên bố quantum
speedup.

### 3.3. Biểu diễn trạng thái ảnh

ViT đóng băng cung cấp:

- CLS query `q`;
- các patch token `t_i`.

Chiếu chúng từ chiều `C=768` xuống không gian trạng thái `d`, mặc định `d=16`:

```text
q_hat = normalize(W q)
z_i   = normalize(W t_i)
```

Trạng thái từ CLS là pure state:

```text
rho_cls = q_hat q_hat^T
```

Patch state là hỗn hợp có trọng số attention theo CLS:

```text
a_i       = softmax(z_i^T q_hat / sqrt(d))
rho_patch = sum_i a_i z_i z_i^T
```

Mixed image state cuối cùng:

```text
rho_x = eta * rho_cls + (1 - eta) * rho_patch
```

Trong cấu hình mặc định, `eta = qsd_cls_mix = 0.5`. Ma trận được đối xứng hóa và
chuẩn hóa để có trace bằng 1.

### 3.4. Biểu diễn trạng thái prompt

Với prompt key `k_j`, factor đầu tiên là:

```text
f_j,1 = normalize(W k_j)
```

Mỗi prompt còn có `R-1` factor học được. Mặc định `R=qsd_rank=4`. Density state
của prompt `j` là:

```text
sigma_j = normalize_trace(sum_r f_j,r f_j,r^T)
```

`R=1` tạo pure prompt state và được dùng làm ablation.

### 3.5. PGM/POVM và Born probabilities

Với `M` prompt và prior đều nhau:

```text
Gamma = (1/M) * sum_j sigma_j + eps * I
E_j   = Gamma^(-1/2) * (sigma_j/M) * Gamma^(-1/2)
```

Born score của prompt `j` đối với ảnh `x`:

```text
b_j = Tr(E_j rho_x)
p_qsd(j|x) = b_j / sum_l b_l
```

Phép nghịch đảo căn bậc hai được tính bằng eigendecomposition của ma trận
`d x d`; với `d=16`, chi phí nhỏ so với ViT.

### 3.6. Ghép với cosine L2P

Cấu hình chính không loại cosine ngay lập tức. Routing logits là:

```text
r_j = cosine(q, k_j) / tau + lambda * log p_qsd(j|x)
```

Trong đó:

- `tau = qsd_cosine_tau`, mặc định `0.1`;
- `lambda` là scalar học được;
- `lambda` khởi tạo bằng `0`.

Vì `lambda=0` và chia cosine cho một hằng số dương không đổi thứ hạng, top-`K`
ban đầu giống chính xác cosine L2P. `lambda` trong implementation là hệ số có
dấu; giá trị âm là tín hiệu chẩn đoán rằng QSD evidence đang phản tương quan với
retrieval hữu ích.

### 3.7. Gradient qua top-K

Top-K rời rạc không truyền gradient. Nhánh dùng straight-through scaling trên
prompt đã chọn:

```text
scale = 1 + p_selected - stop_gradient(p_selected)
P_selected' = scale * P_selected
```

Forward luôn có `scale=1`, nên không đổi độ lớn prompt. Backward vẫn truyền tín
hiệu classification loss về routing probability và QSD parameters.

### 3.8. Measurement retention qua task

Trong khi học task `t`, router cộng dồn:

- mean image density state;
- mean QSD measurement distribution.

Sau task, chỉ hai thống kê nhỏ này được giữ lại; không lưu ảnh và không lưu ViT
embedding 768 chiều.

Ở task sau, POVM hiện tại đo lại các density state cũ. Loss retention là:

```text
L_ret = mean_t KL(p_old_t || p_current_t)
```

Loss tổng:

```text
L = L_L2P + qsd_retention_coeff * L_ret
```

Mặc định `qsd_retention_coeff=0.1`.

### 3.9. Tham số và diagnostics

Với pool 10, `d=16`, rank 4 và ViT-B/16, router bổ sung khoảng 12.8K tham số:

- projection `768 x 16`;
- `10 x 3 x 16` extra prompt factors;
- một residual strength scalar.

Các diagnostics:

| Tên | Ý nghĩa |
|---|---|
| `QSDEnt` | Entropy Born distribution; quá cao và bất biến có thể là uniform collapse |
| `QSDPur` | Purity `Tr(rho_x^2)` của image state |
| `QSDStr` | Hệ số `lambda` đã học |
| `RouteEnt` | Entropy sau khi kết hợp cosine và QSD |
| `QSDRet` | Measurement-retention KL |

### 3.10. Cấu hình chạy chính

```bash
--prompt_router qsd \
--qsd_state_dim 16 \
--qsd_rank 4 \
--qsd_eps 1e-4 \
--qsd_cls_mix 0.5 \
--qsd_cosine_tau 0.1 \
--qsd_memory_size 10 \
--qsd_retention_coeff 0.1
```

### 3.11. Ablation bắt buộc

| Thí nghiệm | Cấu hình | Câu hỏi |
|---|---|---|
| L2P | `--prompt_router cosine` | Baseline trong cùng codebase |
| QSD đầy đủ | `--prompt_router qsd` | Hiệu quả tổng thể |
| Không retention | thêm `--qsd_retention_coeff 0` | Lợi ích có đến từ giữ phép đo? |
| Pure prompt state | thêm `--qsd_rank 1` | Mixed prompt state có cần thiết? |
| CLS only | thêm `--qsd_cls_mix 1.0` | Patch mixture có đóng góp? |
| QSD only | thêm `--qsd_no_cosine_prior` | QSD có tự retrieval được không? |

### 3.12. Tiêu chí kết luận

Có tín hiệu tích cực khi:

1. Final Avg Acc tăng và/hoặc Forgetting giảm qua nhiều seed.
2. `QSDStr` rời 0 một cách ổn định.
3. QSD đầy đủ hơn rank-1 và CLS-only.
4. Bỏ retention làm forgetting xấu đi.

Không đủ bằng chứng khi QSD chỉ ngang baseline, `QSDStr` luôn gần 0, hoặc Born
distribution gần uniform và bất biến.

---

## 4. Nhánh `phase-interference-l2p`: Prompt-Conditioned Phase Interference

### 4.1. Git và phạm vi

```text
Branch: phase-interference-l2p
Base: main
Commit:
  6bb5a44 Add isolated phase interference ablations to L2P
```

`git log main..phase-interference-l2p` chỉ có commit trên. Nhánh không chứa QSD
router, QSD retention hay QSD CLI flags.

### 4.2. Giả thuyết

L2P dùng query toàn ảnh để lấy prompt nhưng không cho prompt thay đổi cách các
vùng ảnh tương tác trước backbone có prompt.

Nhánh kiểm tra giả thuyết:

> Prompt đã được L2P chọn có thể gán pha cho các vùng ảnh; mạch trộn biến chênh
> lệch pha giữa các vùng thành patch weights phụ thuộc toàn cục, qua đó cải thiện
> stability/plasticity hơn một bộ reweighting MLP cùng dung lượng.

Điểm cần chứng minh bằng ablation là lợi ích riêng của phase/interference, không
chỉ lợi ích chung của patch reweighting.

### 4.3. Vị trí can thiệp

Luồng forward:

```text
ảnh
  -> patch_embed của student ViT
  -> cosine retrieval nguyên bản của L2P
  -> top-K prompt nguyên bản
  -> optional patch transform                 [điểm mới duy nhất]
  -> concat(selected prompts, transformed patches)
  -> prompted ViT blocks
  -> classifier
```

Với `--patch_transform none`, không có patch-transform module nào được khởi tạo,
không có tham số mới và không tiêu thụ thêm RNG. Đây là L2P baseline trong cùng
nhánh.

### 4.4. Pool patch thành 16 vùng

ViT-B/16 ở ảnh 224 tạo lưới patch `14 x 14`. Patch embeddings được reshape về
không gian và adaptive-average-pool thành `4 x 4`:

```text
E in R^(196 x 768) -> V = {v_0, ..., v_15}
```

Các prompt đã chọn có shape `K x L x C`; vector điều kiện là:

```text
P_bar = mean(P_selected, dimensions=(K, L))
```

### 4.5. Prompt-conditioned phase encoding

Trước khi chiếu, `v_j` và `P_bar` được layer-normalize không tham số. Hai phép
chiếu `A` và `B` đưa chúng xuống latent dimension `h`, mặc định `h=32`:

```text
u_j = A LayerNorm(v_j)
c   = B LayerNorm(P_bar)
```

Pha của vùng `j`:

```text
phi_j = pi * tanh(u_j^T c / sqrt(h))
```

Trạng thái 4 qubit ban đầu:

```text
|psi> = (1/4) * sum_(j=0)^15 exp(i phi_j) |j>
```

Mỗi biên độ ban đầu có magnitude `1/4`, do đó tổng xác suất bằng 1. Chỉ pha
tương đối khác nhau giữa các vùng.

### 4.6. Mạch trộn `U_theta`

Mạch mặc định có hai layer. Mỗi layer gồm:

1. `RY(theta_y)` trên từng qubit;
2. `RZ(theta_z)` trên từng qubit;
3. CNOT ring: `0->1`, `1->2`, `2->3`, `3->0`.

Đầu ra:

```text
p_j = |<j| U_theta |psi>|^2
```

Implementation lưu phần thực và ảo bằng hai tensor thực thay vì complex dtype,
để tương thích ổn định với PyTorch 1.12.1 CUDA trên Kaggle. Đây vẫn là mô phỏng
đúng các phép quay unitary và permutation CNOT.

Góc mạch được khởi tạo nhỏ nhưng khác 0 (`std=0.1`). Nếu mọi phép trộn là identity,
magnitude đầu ra vẫn đều và phase không thể ảnh hưởng patch weights.

### 4.7. Điều chế patch

`p_j` được reshape thành `4 x 4`, nội suy bilinear về `14 x 14`, rồi áp dụng:

```text
E_i' = E_i * [1 + alpha * tanh(16 * p_tilde_i - 1)]
```

`alpha` được tham số hóa bằng sigmoid:

```text
alpha = alpha_max * sigmoid(alpha_logit)
```

Mặc định:

- `alpha_init=0.05`;
- `alpha_max=0.2`.

Do đó transform khởi đầu nhỏ và không thể tăng vô hạn.

### 4.8. Bốn mode ablation

| Mode | Cơ chế | Vai trò |
|---|---|---|
| `none` | Không khởi tạo transform | L2P gốc |
| `phase` | Prompt-patch phase + mạch 4 qubit | Phương pháp đầy đủ |
| `mlp` | MLP nhận `[v_j; P_bar]`, softmax ra 16 weights | Control cổ điển cùng dung lượng |
| `phase_no_encoding` | Cùng mạch nhưng đặt toàn bộ `phi_j=0` | Kiểm tra phase encoding có cần thiết? |

Với ViT-B/16, latent 32, depth 2:

```text
phase parameters = 49,169
MLP parameters   = 49,218
difference       < 0.1%
```

`phase_no_encoding` vẫn giữ graph edge zero-gradient tới hai projection để DDP
không xem chúng là unused parameters. Forward phase vẫn chính xác bằng 0.

### 4.9. Batchwise và per-image prompt

Config CIFAR-100 gốc dùng `batchwise_prompt=True`. Khi giữ mặc định, transform
được điều kiện bởi prompt đã qua majority selection của batch. Điều này cho phép
so trực tiếp với baseline L2P đã chạy trước đó.

Để kiểm tra đúng giả thuyết prompt riêng cho từng ảnh, dùng:

```bash
--no_batchwise_prompt
```

Khi dùng cờ này, phải chạy lại cả bốn mode. Không so một run per-image với
baseline batchwise vì khi đó hai biến đã thay đổi đồng thời.

### 4.10. Loss

Nhánh không thêm loss phụ:

```text
L = L_L2P
```

Classification loss tự truyền gradient qua patch scaling, probabilities, mạch
và phase projections.

### 4.11. Diagnostics

| Tên | Ý nghĩa | Mốc tham chiếu |
|---|---|---|
| `PatchEnt` | Entropy chuẩn hóa của 16 probabilities | `1` là uniform |
| `PatchPeak` | Mean xác suất vùng lớn nhất | Uniform là `1/16 = 0.0625` |
| `PatchDev` | Mean `abs(scale - 1)` | Gần 0 nghĩa là transform gần như không tác động |
| `PatchAlpha` | Alpha đã học | Khởi tạo `0.05`, bị chặn dưới `0.2` |
| `PhaseStd` | Độ phân tán pha giữa 16 vùng | Phải bằng 0 ở `phase_no_encoding` |

### 4.12. Cấu hình chạy

```bash
--patch_transform phase \
--phase_latent_dim 32 \
--phase_circuit_depth 2 \
--phase_alpha_init 0.05 \
--phase_alpha_max 0.2 \
--phase_circuit_init_std 0.1
```

Thay `phase` lần lượt bằng `none`, `mlp`, `phase_no_encoding` để tạo ma trận
ablation.

### 4.13. Tiêu chí kết luận

Có bằng chứng cho phase/interference khi:

1. `phase` hơn `none` về Final Avg Acc và/hoặc Forgetting.
2. `phase` hơn `mlp` có số tham số tương đương.
3. `phase` hơn `phase_no_encoding`.
4. Kết quả giữ được qua nhiều seed.
5. `PhaseStd`, `PatchPeak` và `PatchDev` cho thấy module không collapse.

Nếu `phase` chỉ hơn `none` nhưng không hơn `mlp`, kết luận hợp lý là patch
reweighting hữu ích; chưa có bằng chứng rằng phase interference mang lại lợi ích
riêng. Nếu `PatchDev` gần 0, module đã học gần identity và run không kiểm tra được
giả thuyết một cách mạnh.

---

## 5. Quan hệ giữa hai nhánh

Hai phương pháp trực giao về vị trí can thiệp:

```text
frozen ViT query
       |
       v
[retrieval]  <--- qsd-prompt thay đổi tại đây
       |
 selected prompts
       |
       v
[patch transform] <--- phase-interference-l2p thay đổi tại đây
       |
 prompted ViT
       |
 classifier
```

Thứ tự nghiên cứu khuyến nghị:

1. Đánh giá từng nhánh độc lập với baseline riêng.
2. Hoàn thành ablation để xác định thành phần có hiệu quả.
3. Chỉ khi cả hai nhánh độc lập đều có tín hiệu, mới tạo thí nghiệm kết hợp.

Nhánh `phase-interference-prompt` là prototype kết hợp kế thừa QSD. Không dùng
nhánh đó cho bảng ablation sạch; dùng `phase-interference-l2p`.

## 6. Giao thức thực nghiệm chung

### 6.1. Giai đoạn sàng lọc

Chạy một seed (`10961`) cho các cấu hình chính:

```text
QSD branch:
  cosine
  qsd

Phase branch:
  none
  mlp
  phase_no_encoding
  phase
```

Không nên kết luận tính hiệu quả từ một seed; đây chỉ là bước loại các cấu hình
rõ ràng không học hoặc collapse.

### 6.2. Giai đoạn xác nhận

Với cấu hình có tín hiệu, chạy ít nhất ba seed giống nhau giữa mọi method. Báo
cáo mean và standard deviation của:

- Final Average Accuracy;
- Average Incremental Accuracy;
- Forgetting;
- Backward Transfer;
- mean old-task accuracy;
- new-task accuracy;
- thời gian huấn luyện và số tham số.

### 6.3. Kết quả tự động

Cả hai nhánh ghi:

```text
<output_dir>/results_summary.json
```

File được cập nhật sau từng task và chứa:

- `final_avg_acc`;
- `avg_incremental_acc`;
- `forgetting`;
- `backward_transfer`;
- `final_old_task_acc`;
- `final_new_task_acc`;
- `final_old_new_gap`;
- `final_per_task_acc`;
- `accuracy_matrix`;
- diagnostics riêng của method;
- tổng thời gian khi run hoàn thành.

## 7. Bảng kết quả đề xuất

### 7.1. QSD retrieval

| Method | Final Acc | AIA | Forgetting | BWT | Old Acc | New Acc | QSDStr | QSDRet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L2P cosine | | | | | | | - | - |
| QSD full | | | | | | | | |
| QSD no retention | | | | | | | | 0 |
| QSD rank 1 | | | | | | | | |
| QSD CLS only | | | | | | | | |

### 7.2. Phase-interference patches

| Method | Final Acc | AIA | Forgetting | BWT | Old Acc | New Acc | PatchDev | PhaseStd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L2P (`none`) | | | | | | | - | - |
| Matched MLP | | | | | | | | 0 |
| Circuit, no phase encoding | | | | | | | | 0 |
| Phase interference | | | | | | | | |

## 8. Giới hạn cần ghi rõ

1. Cả hai phương pháp hiện được mô phỏng trên GPU cổ điển; không có tuyên bố
   lợi thế tốc độ phần cứng lượng tử.
2. Kết quả unit test chỉ xác nhận toán học, gradient và wiring của code; không
   chứng minh accuracy tốt hơn.
3. Tính mới tuyệt đối vẫn cần systematic literature review tại thời điểm viết
   bài.
4. Một seed hoặc chênh lệch nhỏ hơn độ lệch chuẩn không đủ để kết luận.
5. So sánh ablation chỉ hợp lệ khi dataset order, seed, epochs, batch size,
   prompt pool, top-K và batchwise/per-image setting giống nhau.

## 9. File chính

### `qsd-prompt`

```text
prompt.py             mixed states, PGM/POVM, hybrid retrieval, state memory
vision_transformer.py truyền frozen patch tokens vào router
engine.py             retention loss, consolidation, diagnostics
result_summary.py     JSON summary
tests/                 density-state, parity và gradient tests
```

### `phase-interference-l2p`

```text
patch_transform.py             phase encoder, 4-qubit circuit, MLP controls
prompt.py                      điểm chèn sau top-K L2P
vision_transformer.py          truyền cấu hình transform
engine.py                      patch diagnostics và task metrics
result_summary.py              JSON summary
tests/test_phase_ablation.py   baseline parity, unitary probability, gradient,
                               parameter matching và summary tests
```
