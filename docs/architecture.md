# MultiD4RT：模型结构与维度

[返回 README](../README.md) · 实现：[model.py](../multid4rt/model.py)、[config.py](../multid4rt/config.py)

本文对应当前分支的实际实现。新模型采用全部观察 KV 和片段相对时间编码；旧 checkpoint 的计算语义兼容行为见最后一节。

## 1. 输入约定

设 `B` 为 batch，`C` 为本片段相机数，`T` 为观测时刻数，完整同步输入有 `S=C*T` 个观察。`Q` 是选择解码的源图数，`Tq` 是请求输出的目标时刻数，`U` 是目标时刻与源时刻的并集长度。`H,W` 是补边后的大小，`p=14`，`Hp=H/p`，`Wp=W/p`，`L=Hp*Wp`，`D` 为 encoder 宽度。

| 输入 | 维度 / 类型 | 含义 |
|---|---|---|
| `images` | `[B,S,3,H,W]`，float `[0,1]` 或 uint8 | RGB，内部执行 ImageNet 归一化 |
| `camera_id` | `[B,S]`，整数 | 同一物理相机共享非负 ID；不是外参或 embedding 表索引 |
| `time_idx` | `[B,S]`，整数 | 同步相机共享帧标签，保留抽帧间隔 |
| `observation_valid` | `[B,S]`，bool，可选 | 缺失/补齐观察不参与 KV；观察 0 必须有效 |
| `query_time_idx` | `[B,Tq]` 或 `[Tq]`，可选 | 默认返回所有观测时刻；每个请求时刻至少有一张有效输入图 |
| `source_obs_idx` | `[B,Q]` 或 `[Q]`，可选 | 源观察的下标，默认所有观察；不是相机 ID，也不是像素编号 |

例如 `C2T3`，按相机优先展平：

```text
观察下标         0    1    2    3    4    5
相机 camera_id  7    7    7    9    9    9
时间 time_idx   0    5   10    0    5   10
```

帧号 `5` 是标签，通过查表得到张量槽位，不能直接作为长度为 3 的时间轴下标。没有 timestamp 时直接使用共享整数帧号；不要将每台相机的展平序号当成时间。网络支持同片段中的缺失视角；当前任务假定同步数据，不做时钟偏移估计。

输入相机可以移动：同 ID 只表示设备身份，不强制相机在所有时刻的外参相同。ID 在每个片段内有效即可，不需要全数据集统一的相机编号。观察 0 决定参考角色，重排其他观察并同步重排元数据，不改变对应输出的含义。

`H,W` 必须是 14 的倍数且均至少 28；图像由调用方保持比例准备并补边，网络本体不偷偷 resize；forward 不需要已知内外参。

## 2. 共享 encoder：相机和时间在这里怎样加入

源码：[encoder.py](../multid4rt/encoder.py)、[embeddings.py](../multid4rt/embeddings.py)、[attention.py](../multid4rt/attention.py)。

giant 使用 `D=1536`、40 层、24 heads、patch14。前 13 层（索引 0–12）按每张图执行 local attention，初始 patch embedding 为 `[B,S,L,D]`，含 CLS 时为 `[B,S,L+1,D]`。空间位置继承 4RC/DINO 的插值位置编码。

在索引 13 插入融合所需的信息：

1. 将 CLS 替换为 camera token。观察 0 使用 reference token，其余使用共享 context token。这两个 token 表示参考角色，**不是 C 个物理相机的 ID embedding**。
2. 插入一枚共享可学习的 time token。每观察 token 序列为 `[camera_token,time_token,patch_1,...,patch_L]`，尺寸 `[B,S,L+2,D]`。
3. 将该观察的帧编码加到整个 token 序列，每个 token 都收到同一帧编码。

对 `i=0,...,D/2-1`：

```text
ω_i       = 100^(-i/(D/2))
PE(t)     = concat(sin(t*ω), cos(t*ω))           [D]
o         = min(全部有效观察的 time_idx)       每个 batch 元素独立
e_time(t) = W_encoder PE(t-o)                  [D]
Z_s       = Z_s + e_time(time_idx[s])           [L+2,D]
```

`W_encoder` 是独立的 `D→D` 无 bias 线性层，零初始化；频率不作为 checkpoint 参数，没有固定最大时间长度表。输入、GT 索引及输出仍保留原帧号。`time_origin="clip_start"` 只在编码处减去同一个片段原点，例如 `[101,106,111]→[0,5,10]`，保留采样间隔，也不解释为秒。无效补位不参与原点计算；所有相机共用原点，改变查询子集也不改变它。旧模型的 `absolute` 模式使用 `o=0`。

从第 13 层起，奇数索引是 global attention、偶数索引是 local attention：

```text
local： [B*S, L+2, D]
global：[B, S*(L+2), D]
```

global 同时融合所有相机、所有时刻。局部块使用 patch 网格 2D RoPE；global 继承 4RC 的 `pos_nodiff` 位置策略，不能将其描述成额外的相机位姿编码。

**相机 ID 进入 attention logits：** 对每层每个 head 有一个可学习标量 `α_h`，初始化为 0：

```text
logit(i,j,h) = dot(q_i,k_j)/sqrt(d_head)
               + α_h * 1[camera(i)==camera(j) 且 ID>=0]
               + invalid_key_mask(j)
```

同相机可获得偏好，但异相机仍然可被访问。相机 ID 任意一一重命名都保持关系不变，支持 C1、C2、C10 等可变数量；这里没有依赖固定相机编号的 learned embedding 表。该关系在 encoder global attention 中作用于每观察的所有 token。

输出 tap 为全局层索引 `19/27/33/39`，每个 tap 包含：

```text
patch  [B,S,L,2D] = concat(最近 local 特征, 当前 global 特征)
camera [B,S,2D]
time   [B,S,2D]
```

四 tap 几何分支用 `2D`；运动分支取最后 `D` 维，即 global 半。

## 3. 几何分支：从 RGB 到基础点

源码：[geometry.py](../multid4rt/geometry.py)、[DualDPT](../multid4rt/vendor/heads/dualdpt.py)。

四组 `[B,S,L,3072]` 分别 LayerNorm、转为空间网格，经 1×1 卷积投影到 `256/512/1024/1024` 通道，空间尺度依次为 `4Hp×4Wp`、`2Hp×2Wp`、`Hp×Wp`、`ceil(Hp/2)×ceil(Wp/2)`。DPT 逐级融合，特征宽度 256，最终 neck 为 128。

深度与射线有独立的 fusion head：

- 深度 neck 插值回 `H×W`，卷积 `128→32→2`，输出 `depth=exp(z)`、`depth_conf=1+exp(c)`，均 `[B,S,H,W]`。
- ray 辅助分支输出 6 维世界射线信息及 confidence。四尺度为 `Hp×Wp`、`2Hp×2Wp`、`4Hp×4Wp`、`8Hp×8Wp`；`ray` 是最后尺度 `[B,S,8Hp,8Wp,6]`，**不是原图大小**，下游使用时需要显式对齐分辨率。

相机头使用最后 tap 的 camera token `[B,S,3072]`：

```text
pose11 [B,S,11] = [平移3, quaternion_xyzw4, fov_h, fov_w, principal_dx, principal_dy]
K      [B,S,3,3]
c2w    [B,S,4,4]
```

相机是 zero-skew pinhole，假定图像已经去畸变；不预测畸变系数。四元数先归一化。内参解码为：

```text
fx = (W/2) / tan(fov_w/2)     fy = (H/2) / tan(fov_h/2)
cx = W/2 + W*principal_dx    cy = H/2 + H*principal_dy
```

主点 offset 新增线性头零初始化，原 4RC 的 9 维头保留。每张观察独立输出 `K,c2w`，没有硬性跨时刻共享 K，也没有将预测的第一个 c2w 硬编码为单位阵；输入参考角色与 checkpoint 决定参考坐标约定。

基础点直接可微反投影：

```text
x_cam(u,v) = depth(u,v) * [(u-cx)/fx, (v-cy)/fy, 1]
P_s(u,v)   = R_c2w(s) x_cam(u,v) + t_c2w(s)
P          [B,S,H,W,3]
```

基础点采用 depth+K+c2w，ray 是辅助射线输出。轨迹与深度、相机头及 encoder 共用同一计算链。几何回归和相机计算在 float32 中执行。

## 4. 运动分支：全部观察 KV + 时间条件

源码：[metadata.py](../multid4rt/metadata.py)、[motion.py](../multid4rt/motion.py)。

### 全时刻 KV

每个 tap 的所有 global patch 直接展平为：

```text
memory         [B,S*L,D]
memory_camera  [B,S*L]
memory_valid   [B,S*L]
```

其中 `S` 包含**全部输入观察**，不随 `query_time_idx` 子集改变。即使只请求 t10、源图在 t0，t5 的图仍直接进入 KV。camera/time token 不作为额外 KV 元素，patch 已在 encoder 接受时间编码和全局融合。

`metadata` 另建立目标/源时间并集以及 `[B,U,C]` 的观察索引，用于条件聚合、源时间锚定和可见性。这个同刻分组不会缩减新模式的 cross-attention KV。

### 目标时间条件

选定源观察 `s`，其帧号为 `t_s`。第 `m` 个 tap 的目标条件：

```text
τ_m(t)  = mean_valid_camera(global_time_token_m(camera,t))  [B,U,D]
h_m(s,t)= τ_m(t) + W_target PE(t-o) + W_delta PE(t-t_s)     [B,U,D]
```

`W_target/W_delta` 跨 tap、跨源共享，彼此独立，零初始化。不同 tap 的 `τ_m` 不同；source 时间不存在于请求目标中时，内部补上再解码，以计算锚定。

条件既初始化每个目标的 prefix token，也持续调制每轮 AdaLN。AdaLN 一直使用原始条件 `h`，不会被更新后的 prefix 覆盖。

### 共享四轮 decoder

对每个源图、每个 tap，取源 patch `[B,L,D]`，向 U 个目标复制，拼上对应 prefix：

```text
Q0                     [B,U,L+1,D]
cross-attention queries [B,U*(L+1),D]
cross-attention KV      [B,S*L,D]
self-attention queries  [B*U,L+1,D]
```

同一目标内部的 prefix+空间 patch 做 self-attention；不同目标不在 self-attention 中混在一起。仅在 cross-attention 的查询轴合并 U，使 K/V 每轮只按 B 投影一次。它与为每个目标复制同一份 KV 的结果/梯度等价，避免 U 倍 KV 复制和投影；attention 本身仍需处理 `U*(L+1)*S*L` 的相关性计算。

每轮顺序是 `CrossBlock（cross-attention + MLP）→ AdaLNBlock（self-attention + MLP）`，共 4 轮。四个 tap 使用**同一个**完整 decoder，而不是每 tap 对应其中一轮。

AdaLN 使用 `SiLU(h)→Linear(D,6D)` 得到两组 shift/scale/gate：

```text
z = LN(x)*(1+scale_attention) + shift_attention
x = x + gate_attention * SelfAttention(z)
z = LN(x)*(1+scale_mlp) + shift_mlp
x = x + gate_mlp * MLP(z)
```

decoder cross-attention 同样加入每轮每 head 的同相机偏置；源 patch 携带源相机 ID，KV patch 携带对应输入相机 ID。prefix 的 query ID 为 `-1`，相机偏置为零。跨相机仍参与 attention，不额外堆叠一个独立 camera-attention decoder。空间 RoPE 使用源网格和各 KV 图网格，各相机/时刻复用同一空间坐标网格。

每个 tap 解码结果 `[B,U,L+1,D]`；移除 prefix 后交给 MotionDPT。

## 5. 基础点 + 位移以及可见性

MotionDPT 的四路输入宽度为 `D`，采用 `256/512/1024/1024` 投影和 256 宽融合，neck 为 128；插值回 `H×W` 后 `128→32→4`，输出三维场 `g` 和运动置信度：

```text
g(a)      = expm1(a) (a>=0)，-expm1(-a) (a<0)
confidence= 1+exp(raw_confidence)
Δ_s(t)    = g_s(t)-g_s(t_s)
X_s(t)    = P_s+Δ_s(t)
```

指数在 float32 计算，运动头只对溢出量级 logits 设置上限。锚定在激活后进行，因此源时刻严格 `Δ_s(t_s)=0`、`X_s(t_s)=P_s`。最后返回请求的 Tq 个目标，内部补齐的锚时刻不额外输出。

VisibilityHead 对每个目标相机投影 `X_s(t)`，采样目标特征、预测深度/置信度，拼接：源 motion 特征 128 + 目标 patch 投影特征 128 + 4 个几何 cue（归一化投影坐标2、相对深度残差1、log-depth-confidence1）。共享 MLP `260→128→64→1` 输出 logit。

投影坐标及几何 cue 停止梯度，源/目标特征保留梯度。输出 `visibility=sigmoid(logit)*projection_valid`；投影无效位置由独立的 `projection_valid` 标记。`motion_confidence` 是回归可靠性，与遮挡概率不同。

## 6. 输出与下游接口

| 输出 | 维度 |
|---|---|
| `depth` / `depth_conf` | `[B,S,H,W]` |
| `K` / `c2w` / `camera_encoding` | `[B,S,3,3]` / `[B,S,4,4]` / `[B,S,11]` |
| `base_points` | `[B,S,H,W,3]` |
| `trajectory` / `displacement` | `[B,Q,Tq,H,W,3]` |
| `motion_confidence` | `[B,Q,Tq,H,W]` |
| `visibility` / `visibility_logits` / `projection_valid` | `[B,Q,Tq,C,H,W]` |
| `output_camera_id` / `output_camera_valid` | `[B,C]`；按 ID 升序，补齐项无效 |
| `target_observation_valid` | `[B,Tq,C]` |
| `source_obs_idx` / `source_time_idx` / `source_valid` | `[B,Q]` |
| `query_time_idx` | `[B,Tq]` |

`ray/ray_conf/ray_list/ray_conf_list` 为上述低分辨率辅助输出。未提供 visibility 时，对应三个输出键不返回。

```python
from multid4rt.tasks import scene_flow, sample_tracks

# 编码可在同一次推理的不同源/目标请求间复用。
with torch.inference_mode():
    scene = model.encode(images, camera_id, time_idx)
    pred = model.decode(scene, query_time_idx=[0,5,10], source_obs_idx=[0,3])
    flow = scene_flow(pred, start_time=0, end_time=10)  # [B,Q,H,W,3]
    tracks = sample_tracks(pred, torch.tensor([[[0,12.,8.], [3,20.,10.]]]))
    # trajectory [B,N,Tq,3]；visibility [B,N,Tq,C]
```

`scene_flow` 是同一个被跟踪物理点的世界位移，不是两个像素网格直接相减得到的 optical flow。源像素始终保留身份；若需要 t0 图像网格上的 flow，应选择 t0 源图。推理输出的场景尺度由 checkpoint 决定，未知内外参的 RGB 重建不保证绝对米制尺度。

`EncodedScene` 缓存只对应产生它的输入；图片或元数据变化时需要重新编码。

## 7. 一个具体 giant 维度例子

取 `B=1,C=4,T=Tq=8,Q=4`，原始 Kubric 图 `256×256` 补成 `266×266`，于是 `Hp=Wp=19,L=361,U=8,S=32`：

| 位置 | 形状 |
|---|---|
| RGB | `[1,32,3,266,266]` |
| 融合阶段 token | `[1,32,363,1536]` |
| encoder global 序列 | `[1,11616,1536]` |
| 每个几何 tap | `[1,32,361,3072]` |
| 每个运动 tap / all-observation KV | `[1,32,361,1536]` / `[1,11552,1536]` |
| 单源单 tap 的 decoder query | `[1,8,362,1536]` |
| cross-attention query / KV | `[1,2896,1536]` / `[1,11552,1536]` |
| self-attention | `[8,362,1536]` |
| motion DPT 四路投影 | `[8,256,76,76]`、`[8,512,38,38]`、`[8,1024,19,19]`、`[8,1024,10,10]` |
| motion DPT 融合后 neck | `[8,128,152,152]`，随后插值至 `266×266` |
| 单源 motion 输出 | `g:[1,8,266,266,3]`、confidence:`[1,8,266,266]` |
| 四源轨迹 | `[1,4,8,266,266,3]` |
| 四源逐相机 visibility | `[1,4,8,4,266,266]` |

padding 像素也会输出预测，使用结果时应排除这些位置。该例用于说明维度，不承诺所有设备都能承载这一布局。

## 8. 性能和断点兼容

- 当前结构按源图、按 tap 调用共享 decoder；不是把 Q 个源图一起构成超大查询。
- encoder 的 global attention 计算规模仍随 `S*(L+2)` 的平方增长。全时刻 KV 更完整，同时增加 decoder attention 工作量。
- CUDA 无 padding、BF16/FP16 且关闭 query 分块时，同相机 bias 等价分解为额外 Q/K 特征，允许 fused SDPA；不保证所有设备、dtype、mask 都使用 FlashAttention。
- `attention_chunk_size=None`、`visibility_chunk_size=None` 关闭两种 query 分块。CLI `--no-query-chunking` 同时设置这两个值。
- `geometry_chunk_size=8` 是 DualDPT 沿观察图批次的独立计算粒度，已变成显式配置；设为 `None` 才让几何 DPT 一次处理全部观察。`--no-query-chunking` 不改变这个设置。它不减少实际处理的像素。
- 配置保留激活检查点字段以兼容原权重；推理 CLI 禁用激活检查点。

新 checkpoint 的 `format_version=2`，config 记录 `decoder_memory`、`time_origin` 与 `geometry_chunk_size`。所有官方入口通过 `config_from_checkpoint()` 恢复：

| 加载方式 | KV 语义 |
|---|---|
| 新建 `MultiD4RTConfig()` / `tiny()` | `all_observations` |
| 原 4RC 权重初始化新模型 | 使用新模型指定配置，默认 `all_observations` |
| 旧 native checkpoint，config 没有 memory 字段 | `target_time`，复现旧输出 |
| 新 native checkpoint | 保存时的 scope |
| 显式 CLI `--decoder-memory ...` 或传入 config | 按指定值切换，属于架构消融/迁移 |

`target_time` 的 memory 是 `[B,U,C*L,D]`，cross-attention 使用 `[B*U,L+1,D]` 对 `[B*U,C*L,D]`。它仍可从 encoder 间接获得时间上下文，但直接 KV 只有当前目标时刻。两种模式的参数键/形状一致，可以加载同一份模型权重；参数能加载不代表迁移后预测数值或指标不变。切换前需重新检查实际显存与输出。

缺少 `time_origin` 的历史 native config 一律恢复为 `absolute`。新建模型默认 `clip_start`，推理只有显式 `--time-origin clip_start` 才迁移旧模型。时间平移不变性经过测试；迁移旧权重后需要重新检查输出。
