# WAV Loudness Normalizer

FastAPI 服务：将 48 kHz / 16-bit PCM / 单声道或立体声（1–10 秒）的 WAV
归一化到目标响度（EBU R128 / BS.1770），并保证真峰值不超过指定上限。
只施加单一静态增益——不压缩、不削波。

## 启动

```bash
docker compose up --build
```

服务地址：**http://localhost:8000**（交互式 API 文档：http://localhost:8000/docs）

本地开发（不用 Docker）：

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload      # http://localhost:8000
pytest                             # 运行测试
```

## 并发与容量

全部音频 POST 接口共享一个**有界进程执行器**：音频计算在子进程池中执行，
HTTP 处理协程在等待结果期间同时监听 ASGI `http.disconnect` 事件。Compose
运行**单个 Uvicorn 进程**，并发度由进程内执行器限制：

| 环境变量 | 默认 | 范围 | 说明 |
|---|---|---|---|
| `AUDIO_WORKERS` | 2 | 1～8 | 同时运行的音频计算数（子进程池大小） |
| `AUDIO_QUEUE` | 4 | 0～32 | FIFO 排队等待的请求数上限 |

行为规则：

- 运行数 + 排队数满额时，新请求**立即返回 `503`** 并带 `Retry-After: 1`，
  不会继续向进程池堆积任务；
- **客户端断开**（关闭 TCP 连接）在排队或计算期间都会被 HTTP 层通过
  `http.disconnect` 感知并取消对应等待任务，不会留下占着名额的死等待：
  - 排队中断开：等待任务**立即退出 FIFO 队列并释放等待位**，后续请求立刻
    可以入队，不再误收 `503`；
  - 运行中断开：**只丢弃结果**，工作位要等子进程里的计算真正结束才释放
    （无法中止已在子进程内运行的计算）；
  - 断开与完成/异常竞态时，先到者胜：计算先完成则照常走响应路径
    （对已关闭的 socket 写入是静默无操作），名额在任一路径下都只释放一次；
- 完成、业务异常（`422`）与取消竞态均只释放一次名额，队列严格 FIFO，
  后续请求不插队；
- `GET /health` 不经过队列，饱和时仍可响应；
- 执行器随应用启动创建、关闭时回收。

> 实现要点：路由不直接 `await executor.run(...)`，而是把它放进一个子任务，
> 与「读取下一个 ASGI 消息」的监听任务 `asyncio.wait(..., FIRST_COMPLETED)`；
> 请求体在路由执行前已被 FastAPI 读完，监听器只会等到 `http.disconnect`。
> 子进程执行器侧的状态机（`queued → running → done`，运行中取消经
> `cancelled` 过渡）保证名额只释放一次。相关测试同时用 `httpx` ASGI 传输
> （取消请求任务）和**真实 Uvicorn + 真实 TCP socket**（真正关闭连接）
> 覆盖两阶段断开与容量恢复。

## API

### `POST /v1/normalize`

请求体（JSON）：

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `wav_base64` | string | 必填 | Base64 编码的 WAV：48 kHz、16-bit PCM、单/双声道、1–10 s |
| `target_lufs` | number | -30 ~ -10 | 目标综合响度 (LUFS) |
| `ceiling_dbtp` | number | -6 ~ -0.5 | 真峰值上限 (dBTP) |

响应（200）：

| 字段 | 说明 |
|---|---|
| `wav_base64` | 处理后的 WAV（与输入同格式），Base64 |
| `input_loudness_lufs` / `output_loudness_lufs` | 输入 / 输出综合响度 (LUFS) |
| `input_true_peak_dbtp` / `output_true_peak_dbtp` | 输入 / 输出真峰值 (dBTP) |
| `gain_db` | 实际施加的静态增益 (dB) |
| `target_reached` | 输出响度与目标误差 ≤ 0.1 LU 时为 `true` |

错误：`422` —— Base64 非法、WAV 格式/时长不合规、静音（或低于响度门限无法测量）、参数越界。

### `POST /v1/compare-gain`

核验两个 WAV 的**重叠内容**是否只差「统一增益 + 整数采样时移」。两份输入都必须是
48 kHz / 16-bit PCM / 1–10 s，且**声道数一致**（单↔双声道直接 `422`）。

请求体（JSON）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `reference_wav_base64` | string | 原件 WAV，Base64 |
| `candidate_wav_base64` | string | 候选 WAV，Base64，格式与声道数需与原件一致 |

响应（200）：

| 字段 | 说明 |
|---|---|
| `offset_samples` | 整数时移，= 候选对应采样索引 − 原件采样索引；正数表示候选晚于原件；搜索范围 ±100 ms（±4800 采样） |
| `gain_db` | 对齐后对全部声道共用的最小二乘有符号倍率的**幅度**（dB） |
| `polarity_inverted` | 有符号倍率为负（反相）时为 `true` |
| `reference_unmatched_samples` | 原件未参与重叠的帧数 |
| `candidate_unmatched_samples` | 候选未参与重叠的帧数 |
| `anomaly_intervals` | 残差超阈窗合并出的半开区间 `[start_ms, end_ms)`，时间轴沿原件 |
| `overlap_gain_only` | 无任何异常区间时为 `true` |

核验流程：

1. 在 ±100 ms 范围内对全部声道计算**重叠能量归一化互相关**。所有偏移的交叉和
   由一次**基分解 FFT 批量**求得（int16 样本两侧各拆成高 8 位与低 8 位，
   共 4 个分量卷积；双声道打包进同一复数 FFT 的实/虚部），各分量按
   Schatzman（1996）的 radix-2 FFT 舍入界
   `8·ε·log₂N·‖p‖₂·‖q‖₂`（全信号能量，满量程 10 s 双声道下最坏约 2.3e-3）
   证明可安全四舍五入为**精确整数**，无法被该界证明的极少数偏移自动回退为
   逐偏移精确整数点积；各偏移重叠段能量由整数累加和给出。因此偏移比较全程为
   64 位/Python 大整数精确运算——两个偏移精确绝对相关相等 iff 真的相等，
   取绝对相关最大者；只有精确值真正并列时才按「绝对偏移最小、再按数值最小」
   取舍，不再依赖任何 FFT 容差带。工作量为 O(N log N)，不随近峰数量增长；
2. 对齐后在**全部声道的整体重叠段**上做一次最小二乘，得到全声道共用的有符号倍率
   `g`：`gain_db = 20·log10|g|`，`g < 0` 即 `polarity_inverted`；
3. 沿原件时间轴按不重叠 100 ms 窗逐窗计算残差
   `candidate − g·reference`（尾窗不足 100 ms 也照算；完全不重叠的窗跳过），
   残差 RMS 超过 `max(2/32768, 原件重叠段 RMS 的 1%)` 的窗标记为异常并合并成
   半开区间返回（毫秒）。

近周期信号的边界行为：

* 周期 100 采样的正弦（幅度约 10000 LSB）叠加单采样 +500 LSB 标记、候选循环
  右移一个周期（100 采样）时，返回 `offset_samples=100`（两侧各 100 个未重叠
  样本），而不是被 FFT 舍入并列的 0；
* **极端近峰但无精确 ±1**：合法 10 秒、48 kHz、16-bit 双声道，两声道均以
  2 采样为周期交替取约 ±10000 LSB，候选仅在中部、一个声道的一个采样改动
  100 LSB。此时 ±100 ms 范围内约 9600 个偏移的绝对相关都极接近 1（差距约
  1e-10 量级）却**没有任何偏移精确等于 1**，旧实现无法提前退出、对每个近峰
  重算整段整数点积（约 24 秒）。现由上述一次批量精确 FFT 统一给出全部偏移的
  精确交叉和（约 0.5 秒），再按各偏移**实际重叠段**的精确绝对相关选出
  `offset_samples=0`；真并列仍按绝对偏移最小、再按数值最小处理（例如纯周期
  信号无标记时 0 与周期倍数偏移并列，选 0）。

错误：`422` —— Base64 非法、任一 WAV 不合规、声道数不符、任一输入全静音（重叠段
无有效能量无法估计增益/时移时同样 `422`）。

### `POST /v1/cut-splice`

按输入采样索引删除若干半开区间，把保留片段按原顺序拼接，并在每处接缝做线性
交叉淡化；可选地携带一组**时间标注**（cue），响应给出每个标注在输出时间轴上
的映射。输入 WAV 校验与 `/v1/normalize` 相同（48 kHz / 16-bit PCM /
单或双声道 / 1–10 s）。

请求体（JSON）：

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `wav_base64` | string | 必填 | Base64 编码的 WAV：48 kHz、16-bit PCM、单/双声道、1–10 s |
| `remove_intervals` | array<[int, int]> | 必填 | 半开删除区间 `[start, end)`（输入采样索引）；须严格升序、互不相接（相邻区间至少隔 1 帧）、不越界；空数组表示不删除 |
| `crossfade_ms` | int | 0 ~ 20 | 接缝淡化时长（毫秒）；0 表示直接拼接 |
| `cue_intervals` | array<object> | 可选，≤ 2000 项 | 输入采样轴上的半开标注区间 `[start, end)`；每项含非空且唯一的字符串 `id` 与整数 `start`/`end`；顺序任意、可互相重叠；省略时不做映射 |

响应（200）：

| 字段 | 说明 |
|---|---|
| `wav_base64` | 拼接后的 WAV（与输入同格式），Base64；允许短于 1 秒 |
| `output_frames` | 输出帧数 = 保留帧总数 − 接缝数 × F |
| `mapped_cues` | 按**请求顺序**给出的标注映射；未提供 `cue_intervals` 时为空数组 |
| `mapped_cues[].id` | 请求中的标注 id |
| `mapped_cues[].fragments` | 该标注与保留片段相交的碎片，按源位置升序；整项被删尽时为空数组 |
| `mapped_cues[].fragments[].source_interval` | 碎片在输入采样轴上的半开区间 |
| `mapped_cues[].fragments[].output_interval` | 同一碎片在输出时间轴上的半开区间 |

处理规则：

1. 删除 `remove_intervals` 后，保留片段按原顺序拼接；
2. 每处接缝将左段末尾与右段开头的 `F = 48 × crossfade_ms` 帧重叠，全部声道
   逐帧按从 0 到 1（含端点）的线性权重混合：第 `i` 帧（0 起）右段权重
   `i/(F-1)`、左段权重 `1 − i/(F-1)`；`F = 0` 时直接拼接；
3. 有接缝时，首/尾保留片段至少 `F` 帧、中间片段至少 `2F` 帧，保证同一采样
   不被两处淡化复用；
4. 标注映射：第 `j` 段保留片段（从 0 计）的输出起点 = 前 `j` 段总帧数 −
   `j × F`，源帧 `x` 落在第 `j` 段 `[s, e)` 内时映到 `x + 输出起点_j − s`。
   每个标注与所有保留片段求交，落在删除段内的部分消失；交叉淡化区两侧的
   碎片会映到**重叠**的输出区间（重叠不超过 F 帧），两侧都原样保留，
   不合并、也不重复扣除 F。映射按「标注排序 + 保留片段单次扫描」完成，
   不逐项遍历全部保留片段。

错误：`422` —— Base64 非法、WAV 不合规、区间为空/越界/乱序/相接、删除后无
保留内容（删空）、淡化容量不足、`crossfade_ms` 越界、标注 id 为空或重复、
标注区间为空/越界、标注超过 2000 项。

### `POST /v1/repair-gaps`

修补已知的短时**零值缺口**：只替换缺口内部的采样，帧数、声道格式与缺口外的
PCM 保持完全一致。输入 WAV 校验与 `/v1/normalize` 相同（48 kHz / 16-bit PCM /
单或双声道 / 1–10 s）。

请求体（JSON）：

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `wav_base64` | string | 必填 | Base64 编码的 WAV：48 kHz、16-bit PCM、单/双声道、1–10 s |
| `gaps` | array<[int, int]> | 必填 | 半开缺口 `[start, end)`（输入采样索引）；严格升序、互不重叠；每段长 1～20 ms（48～960 帧），距首尾至少 40 ms（1920 帧），相邻缺口间隔至少 60 ms（2880 帧），且缺口内所有声道采样必须为零；空数组表示不修补 |

响应（200）：

| 字段 | 说明 |
|---|---|
| `wav_base64` | 修补后的 WAV（与输入同帧数、同格式；缺口外 PCM 逐采样不变），Base64 |
| `repairs` | 每段缺口的修补结果，按输入顺序给出 `start`、`end`、`method` |
| `repairs[].method` | `predicted`（LPC 双向外推交叉融合）或 `linear`（边界连续线性插值回退） |

修补算法（每个声道独立处理，均使用缺口两侧各 30 ms 的**原始**样本）：

1. 对左右上下文分别拟合带正则化的 24 阶一步线性预测（LPC）：
   `x[n] = Σ a[k]·x[n−24+k]`，以信号平均能量为尺度的岭回归最小二乘求解；
2. 左侧预测器向前递归外推缺口；右侧在**反转**的上下文上拟合同样的模型、
   反向递归外推后再反转回来。两路估计按缺口位置线性交叉融合
   （缺口起点全用左估计、终点全用右估计；单采样缺口取二者均值）；
3. 任一侧 30 ms 上下文**全静音**（该声道全零）时无法预测，返回 `422`；
4. 预测出现**非有限值**，或左/右外推及融合结果中任一值**超出 PCM 满量程
   `[-1, 1]`** 时，该声道回退为穿过两个边界样本 `x[start-1]`、`x[end]` 的
   直线插值（边界连续）。只有所有声道都成功预测时该缺口才标记为
   `predicted`，任一声道回退则整段标记为 `linear`；
5. 仅写入缺口内部采样；缺口外样本经同一量化路径逐位保持不变。

错误：`422` —— Base64 非法、WAV 不合规、缺口为空/越界/乱序/重叠、长度不在
1～20 ms、距首尾不足 40 ms、相邻缺口间隔不足 60 ms、缺口内存在非零采样、
某声道任一侧 30 ms 上下文全静音。

### `POST /v1/denoise`

用用户标注的**纯底噪参考段**对整段音频做谱减法降噪：压低全段稳定噪声，
帧数、声道数与格式保持不变。输入 WAV 校验与 `/v1/normalize` 相同
（48 kHz / 16-bit PCM / 单或双声道 / 1–10 s）。

请求体（JSON）：

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `wav_base64` | string | 必填 | Base64 编码的 WAV：48 kHz、16-bit PCM、单/双声道、1–10 s |
| `noise_interval` | array<[int, int]> | 必填 | 半开参考区间 `[start, end)`（输入采样索引），须为纯噪声段；在界内、长 0.2～2 s（9600～96000 帧），且各声道均有非零采样 |
| `strength` | number | 0 ~ 1 | 抑制强度；0 表示不处理（PCM 逐采样不变），1 为完全谱减 |

响应（200）：

| 字段 | 说明 |
|---|---|
| `wav_base64` | 降噪后的 WAV（与输入同帧数、同声道、同格式），Base64 |

处理流程（每个声道独立处理）：

1. 以 1024 点 Hann 窗、256 点步长对全段作 STFT（信号左端补零使每个采样
   恰好被 4 帧覆盖，窗平方重叠和恒为 1.5）；
2. 取**完全落在参考区间内**的各帧，逐频点计算功率中位数，作为噪声功率
   估计 `N`；
3. 对全段每帧每频点按
   `G = max(0.1, sqrt(max(0, 1 − strength × N / (|X|² + 1e-12))))`
   计算增益，并沿时间轴作 3 帧均值平滑（边界处复制端点）；
4. 以平滑后的增益缩放频谱，重叠相加还原（按窗平方和归一化），量化回
   16-bit PCM。`strength = 0` 时增益恒为 1，输出 PCM 与输入逐采样一致。

错误：`422` —— Base64 非法、WAV 不合规、参考区间为空/越界、长度不在
0.2～2 s、某声道参考段全零、`strength` 越界。

### `POST /v1/compress`

前馈式动态范围压缩：检测包络对所有声道施加**同一增益**（立体声像保持
不变），帧数、声道数与格式不变。输入 WAV 校验与 `/v1/normalize` 相同
（48 kHz / 16-bit PCM / 单或双声道 / 1–10 s）。

请求体（JSON）：

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `wav_base64` | string | 必填 | Base64 编码的 WAV：48 kHz、16-bit PCM、单/双声道、1–10 s |
| `threshold_dbfs` | number | -40 ~ -6 | 压缩阈值 (dBFS) |
| `ratio` | number | 1 ~ 20 | 压缩比；1 表示不压缩（PCM 逐采样不变） |
| `attack_ms` | number | 1 ~ 50 | 包络启动时间（毫秒） |
| `release_ms` | number | 10 ~ 500 | 包络释放时间（毫秒） |
| `knee_db` | number | 0 ~ 12 | 软拐点宽度 (dB)；0 为硬拐点 |

响应（200）：

| 字段 | 说明 |
|---|---|
| `wav_base64` | 压缩后的 WAV（与输入同帧数、同声道、同格式），Base64 |

处理流程：

1. 每帧取各声道绝对值的最大值作为检测电平 `p`；包络 `e` 初值为 0，逐帧按
   `e = a·e_prev + (1−a)·p` 更新，其中 `a = exp(−1/(48·m))`（48 kHz 即每毫秒
   48 采样），`p` 高于前一帧包络时 `m = attack_ms`，否则 `m = release_ms`；
2. 令 `L = 20·log10(max(e, 1e-12))`、`T = threshold_dbfs`、`R = ratio`、
   `K = knee_db`，目标电平：`L ≤ T−K/2` 时取 `L`；`L ≥ T+K/2` 时取
   `T + (L−T)/R`；两者之间（软拐点区）取
   `L + (1/R − 1)·(L − T + K/2)²/(2K)`；`K = 0` 为硬拐点，只有前两条分支；
3. 把目标电平与 `L` 的差作为增益（dB）同时施加到所有声道，量化回
   16-bit PCM。增益恒 ≤ 1（只衰减、不提升）；`ratio = 1` 时增益恒为 0 dB，
   输出 PCM 与输入逐采样一致。

错误：`422` —— Base64 非法、WAV 不合规、任一参数越界。

### `POST /v1/time-stretch`

WSOLA 变速不变调：按时长倍率 `1/speed` 改变 WAV 时长而保持原音高。
输入 WAV 校验与 `/v1/normalize` 相同（48 kHz / 16-bit PCM /
单或双声道 / 1–10 s）。

请求体（JSON）：

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `wav_base64` | string | 必填 | Base64 编码的 WAV：48 kHz、16-bit PCM、单/双声道、1–10 s |
| `speed` | number | 0.8 ~ 1.25 | 时间倍率；输出帧数 = `round(输入帧数 / speed)`；1 时 PCM 逐采样不变 |

响应（200）：

| 字段 | 说明 |
|---|---|
| `wav_base64` | 变速后的 WAV（同采样率、位深、声道数），Base64 |
| `output_frames` | 输出帧数 = `round(input_frames / speed)` |

处理规则（WSOLA，波形相似叠加）：

1. 分析片段长 40 ms（1920 帧），常规输出步长 20 ms（960 帧），相邻片段在
   960 帧交叠区做 0→1 / 1→0 的**线性淡化**混合；
2. **首片**固定取输入起点 `[0, 1920)`，**末片**固定取输入末尾
   `[N−1920, N)`；中间片的输出起点依次为 960、1920、…（末片之前）；
3. 每个中间片的**名义输入起点**由其输出位置在首、末片锚点之间
   **线性映射**：`round((N−1920) · 输出位置 / 末片输出位置)`，
   再在名义位置 **±240 帧**内搜索实际起点；
4. 搜索以新片段头部 960 帧与前一选中片段尾部 960 帧的**归一化互相关**
   为准：各声道点积求和后除以双方能量平方根，立体声两声道因此共用同一
   起点；归一化相关精确并列时取**离名义位置最近**、再取**较小索引**；
   参考交叠区零能量（或全部候选头部零能量）时不做相关，直接取最近位置；
5. **末片输出步长按需缩短**（1～960 帧），使输出总帧数恰好为
   `round(输入帧数 / speed)`。末步长不足 960 帧时，前一片段在末片起点
   之后仍延续 `1920 − 末步长` 帧，末片的完整交叠区即为该长度（而非
   末步长）：整个交叠区把已有的混合输出与末片头部按 0→1 线性权重
   淡化（各声道共用同一权重序列），之后末片剩余 PCM 逐采样复制、
   与输入尾部一致。例：144000 帧、`speed=0.8` 时输出 180000 帧，
   末步长 480 帧，交叠区 1440 帧，输出末 480 帧即输入末 480 帧；
6. `speed = 1` 跳过全部浮点淡化运算，输出 PCM 与输入**逐位相同**。

错误：`422` —— Base64 非法、WAV 不合规、`speed` 越界。

### `POST /v1/align-stereo`

双声道对齐：在用户标注的**校准区间**内估计右轨相对左轨的整数采样偏移
（±240 帧），并把**整段右轨**平移对齐到左轨时间轴（左轨逐采样不变）。
输入 WAV 校验与 `/v1/normalize` 相同，但**仅接受双声道**
（48 kHz / 16-bit PCM / 1–10 s）。

请求体（JSON）：

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `wav_base64` | string | 必填 | Base64 编码的 WAV：48 kHz、16-bit PCM、**双声道**、1–10 s |
| `calibration_interval` | array<[int, int]> | 必填 | 半开校准区间 `[start, end)`（输入采样索引），约定两轨同源；在界内、长 0.2～2 s（9600～96000 帧） |

响应（200）：

| 字段 | 说明 |
|---|---|
| `wav_base64` | 对齐后的 WAV（同帧数、双声道、同格式），Base64 |
| `offset_samples` | 整数偏移 = 右轨索引 − 左轨索引；正数表示右轨晚于左轨；搜索范围 ±240 帧 |
| `polarity_inverted` | 最优相关为负（右轨已反相）时为 `true` |
| `correlation` | 校准段在所选偏移处的有符号归一化互相关（保留 6 位小数） |

处理规则：

1. 对每个候选偏移 `d ∈ [-240, 240]`，取区间内两轨的有效重叠段
   （长 `区间长 − |d|`），各自减去自身均值后计算归一化互相关。全部统计量
   基于 16-bit 样本的**精确整数运算**（`cross = n·ΣLR − ΣL·ΣR`、
   `E_L = n·ΣL² − (ΣL)²`、`E_R = n·ΣR² − (ΣR)²`，
   `corr = cross / √(E_L·E_R)`），并列是真并列；
2. 选绝对相关最大者；并列时依次取**绝对偏移最小**、**数值最小**；
3. 重叠段任一侧去均值后零能量（如恒定/静音段）的偏移无效；**无有效候选**
   或**最优绝对相关 < 0.8**（精确判定 `25·cross² ≥ 16·E_L·E_R`）返回 `422`；
4. 左轨逐采样不变；整段右轨按 `out[i] = right[i + d]` 平移，越界帧补零；
   最优相关为负时右轨反相——先升到 32-bit 取负再钳回 16-bit
   （`-32768 → 32767`，不溢出）；
5. 偏移为 0 且正相关时，输出 PCM 与输入逐采样一致。

错误：`422` —— Base64 非法、WAV 不合规或非双声道、区间为空/越界/长度不在
0.2～2 s、区间内无有效候选（任一侧去均值后零能量）、最优绝对相关 < 0.8。

### `POST /v1/eq`

参量均衡：把 1～6 个 peaking 频段按请求顺序**串联**作用到整段音频上调整音色，
滤波后仅施加**全段统一的非正增益**，使 4 倍重采样真峰值不超过上限（不削波）。
输入 WAV 校验与 `/v1/normalize` 相同（48 kHz / 16-bit PCM /
单或双声道 / 1–10 s）。

请求体（JSON）：

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `wav_base64` | string | 必填 | Base64 编码的 WAV：48 kHz、16-bit PCM、单/双声道、1–10 s |
| `bands` | array<object> | 1～6 项 | peaking 频段，按数组顺序依次处理 |
| `bands[].frequency_hz` | number | 40 ~ 16000 | 频段中心频率 (Hz) |
| `bands[].q` | number | 0.3 ~ 10 | 品质因数 Q |
| `bands[].gain_db` | number | -12 ~ 12 | 中心频率处增益 (dB)，负值为衰减 |
| `ceiling_dbtp` | number | -6 ~ -0.5 | 真峰值上限 (dBTP) |

响应（200）：

| 字段 | 说明 |
|---|---|
| `wav_base64` | 均衡后的 WAV（与输入同帧数、同格式），Base64 |
| `input_true_peak_dbtp` / `output_true_peak_dbtp` | 处理前 / 后真峰值 (dBTP)；静音输入为 `null` |
| `post_gain_db` | 实际施加的全段统一后置增益 (dB)，恒 ≤ 0 |

处理规则：

1. 每个频段按 RBJ Audio EQ Cookbook 的 peaking 公式在 48 kHz 下生成 biquad
   系数：`A = 10^(gain_db/40)`，`w0 = 2π·frequency_hz/48000`，
   `alpha = sin(w0)/(2Q)`，`b = [1+alpha·A, −2cos(w0), 1−alpha·A]`，
   `a = [1+alpha/A, −2cos(w0), 1−alpha/A]`，归一化使 `a0 = 1`；
2. 频段按请求顺序串联滤波，每个 biquad 以**零初始状态**作用于整段信号；
   各声道独立处理，同一频段在所有声道上使用相同系数；
3. 滤波后测 4 倍重采样真峰值，对全段施加统一增益
   `post_gain_db = min(0, ceiling_dbtp − 滤波后真峰值)`——只衰减、不提升；
4. 量化回 16-bit PCM 后**重新测量**真峰值，仍超限则继续回退增益
   （最多 10 次），不削波。

错误：`422` —— Base64 非法、WAV 不合规、频段数量不在 1～6、任一频段参数或
`ceiling_dbtp` 越界或为非有限值（NaN/±Inf）。

### `POST /v1/onsets`

谱通量瞬态检测：返回供剪辑定位的瞬态起点（onset）列表，**不修改音频**。
输入 WAV 校验与 `/v1/normalize` 相同（48 kHz / 16-bit PCM /
单或双声道 / 1–10 s）。

请求体（JSON）：

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `wav_base64` | string | 必填 | Base64 编码的 WAV：48 kHz、16-bit PCM、单/双声道、1–10 s |
| `min_relative_flux` | number | 0 ~ 1，默认 0.1 | 候选通量下限，为全片最大通量的倍数 |

响应（200）：

| 字段 | 说明 |
|---|---|
| `onsets` | 检测到的瞬态起点，按时间升序；全静音时为空数组 |
| `onsets[].frame_index` | 窗口中心采样索引，钳到 `[0, n-1]` |
| `onsets[].strength` | 该点通量除以全片最大通量（0, 1] |

检测规则：

1. 各声道用居中 1024 点 Hann 窗、256 帧步长作实数 FFT（信号两端各补零
   512 采样，第 k 帧中心对准采样 `k*256`）；逐帧将本帧与前一帧各频点幅度
   的正差求和作为通量（首帧与全零谱比较），双声道逐帧取较大值；
2. 候选帧须为**严格局部峰**（平台峰取最早帧），通量严格高于前后各 20 帧
   通量的中位数加 3 倍 MAD，且达到全片最大通量的 `min_relative_flux` 倍；
3. 候选按通量降序、同值按时间升序，依次选取与所有已选点至少相隔
   80 ms（3840 采样 = 15 帧）者；
4. 结果按时间升序返回，`strength = 通量 / 全片最大通量`；
   全静音（最大通量为 0）返回空数组。

错误：`422` —— Base64 非法、WAV 不合规、`min_relative_flux` 越界或为
非有限值（NaN/±Inf）。

### `POST /v1/convolution-reverb`

卷积混响：用上传的**脉冲响应**（IR）给音频加混响尾音。FFT 线性卷积，
输出 `N + M − 1` 帧（N 为原音帧数，M 为脉冲帧数）。原音校验与其它端点
相同（48 kHz / 16-bit PCM / 单或双声道 / 1–10 s）；脉冲响应同格式、
长 1–2 s、不能全静音。

请求体（JSON）：

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `wav_base64` | string | 必填 | Base64 编码的 WAV：48 kHz、16-bit PCM、单/双声道、1–10 s |
| `ir_wav_base64` | string | 必填 | Base64 编码的脉冲响应 WAV：48 kHz、16-bit PCM、单/双声道、1–2 s，不能全静音 |
| `wet` | number | 0 ~ 1 | 湿声比例；0 为补零干声，1 为纯卷积 |

响应（200）：

| 字段 | 说明 |
|---|---|
| `wav_base64` | 混响后的 WAV（48 kHz、16-bit PCM、N+M−1 帧），Base64 |
| `output_frames` | 输出帧数 = 原音帧数 N + 脉冲帧数 M − 1 |
| `post_gain_db` | 实际施加的全声道统一增益 (dB)，恒 ≤ 0；未衰减时为 0 |

处理规则：

1. 声道匹配：单声道 IR 作用于原音各声道；双声道 IR 配单声道原音时把干声
   复制到两个输出声道（输出为双声道），配双声道原音时逐声道卷积；
2. 各输出声道 = `(1−wet) × 补零到 N+M−1 帧的干声 + wet × 卷积湿声`，
   卷积由 FFT 线性卷积（`scipy.signal.fftconvolve`，`mode="full"`）计算；
3. 混合后若超出 16-bit PCM 可表示范围 `[−1, 32767/32768]`，对所有声道
   施加同一非正增益再量化回 16-bit PCM，不裁剪；未超限时
   `post_gain_db` 为 0。

错误：`422` —— Base64 非法、任一 WAV 不合规、脉冲响应时长不在 1–2 s、
脉冲响应全静音、`wet` 越界或为非有限值（NaN/±Inf）。

### 处理流程

1. 用 `pyloudnorm`（BS.1770 K 加权）测输入综合响度；
2. 用 `scipy.signal.resample_poly` 做 4 倍过采样，测最高真峰值；
3. 静态增益取 `min(目标响度所需增益, 峰值上限允许的最大增益)`；
4. 施加增益并量化回 16-bit PCM，对量化后的输出**重新测量**；
   若真峰值超限则回退增益后重试（最多 10 次）；
5. 返回同格式 WAV 及全部测量值。

### 调用示例

```bash
# 生成一个 3 秒测试 WAV 并归一化到 -16 LUFS / -1.5 dBTP
python3 - <<'EOF'
import base64, io, json, wave, urllib.request
import numpy as np

t = np.arange(48000 * 3) / 48000
pcm = np.round(0.2 * np.sin(2 * np.pi * 440 * t) * 32768).astype("<i2")
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(48000)
    wf.writeframes(pcm.tobytes())

req = urllib.request.Request(
    "http://localhost:8000/v1/normalize",
    data=json.dumps({
        "wav_base64": base64.b64encode(buf.getvalue()).decode(),
        "target_lufs": -16.0,
        "ceiling_dbtp": -1.5,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req))
print({k: v for k, v in resp.items() if k != "wav_base64"})
with open("normalized.wav", "wb") as f:
    f.write(base64.b64decode(resp["wav_base64"]))
EOF
```

```bash
# 生成两个 2 秒 WAV：候选 = 原件 +6 dB、整体晚 31 个采样，核验是否仅差增益与时移
python3 - <<'EOF'
import base64, io, json, wave, urllib.request
import numpy as np

rate = 48000
t = np.arange(rate * 2) / rate
sig = 0.2 * np.sin(2 * np.pi * 997 * t + 0.3 * np.sin(2 * np.pi * 3.1 * t))
ref = np.round(sig * 32768).astype("<i2")

shift, gain = 31, 10 ** (6.0206 / 20)
cand = np.zeros_like(ref)
cand[shift:] = np.clip(np.round(ref[:-shift].astype(float) * gain), -32768, 32767)

def enc(pcm):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()

req = urllib.request.Request(
    "http://localhost:8000/v1/compare-gain",
    data=json.dumps({
        "reference_wav_base64": enc(ref),
        "candidate_wav_base64": enc(cand),
    }).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.dumps(json.load(urllib.request.urlopen(req)), indent=2))
# => {"offset_samples": 31, "gain_db": 6.021, "polarity_inverted": false,
#     "reference_unmatched_samples": 31, "candidate_unmatched_samples": 31,
#     "anomaly_intervals": [], "overlap_gain_only": true}
EOF
```

```bash
# 生成 3 秒测试 WAV，删除 1.0–1.5 s 与 2.0–2.2 s 两段，接缝淡化 10 ms，
# 并同步 4 条时间标注到剪辑后的时间轴
python3 - <<'EOF'
import base64, io, json, wave, urllib.request
import numpy as np

rate = 48000
t = np.arange(rate * 3) / rate
pcm = np.round(0.2 * np.sin(2 * np.pi * 440 * t) * 32768).astype("<i2")
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
    wf.writeframes(pcm.tobytes())

req = urllib.request.Request(
    "http://localhost:8000/v1/cut-splice",
    data=json.dumps({
        "wav_base64": base64.b64encode(buf.getvalue()).decode(),
        "remove_intervals": [[48000, 72000], [96000, 105600]],
        "crossfade_ms": 10,
        "cue_intervals": [
            {"id": "intro",    "start": 0,      "end": 48000},
            {"id": "verse",    "start": 47000,  "end": 73000},   # 跨越删除段
            {"id": "cut-away", "start": 50000,  "end": 60000},   # 整段被删除
            {"id": "outro",    "start": 110000, "end": 120000},
        ],
    }).encode(),
    headers={"Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req))
# 保留 110400 帧，2 处接缝各重叠 48*10=480 帧 => 109440
print(resp["output_frames"])  # 109440
for cue in resp["mapped_cues"]:
    print(cue["id"], cue["fragments"])
# intro    [{'source_interval': [0, 48000], 'output_interval': [0, 48000]}]
# verse    [{'source_interval': [47000, 48000], 'output_interval': [47000, 48000]},
#           {'source_interval': [72000, 73000], 'output_interval': [47520, 48520]}]
#          # 两片输出区间在交叉淡化区重叠 480 帧（F），分别保留不合并
# cut-away []                        # 整项删尽，碎片为空
# outro    [{'source_interval': [110000, 120000], 'output_interval': [75440, 85440]}]
with open("spliced.wav", "wb") as f:
    f.write(base64.b64decode(resp["wav_base64"]))
EOF
```

```bash
# 生成 2 秒 440 Hz 正弦，在 1.000～1.005 s 与 1.500～1.510 s 挖两处零值缺口后修补
python3 - <<'EOF'
import base64, io, json, wave, urllib.request
import numpy as np

rate = 48000
n = rate * 2
t = np.arange(n) / rate
pcm = np.round(0.5 * np.sin(2 * np.pi * 440 * t) * 32768).astype("<i2")

gaps = [[48000, 48240], [72000, 72480]]  # 5 ms、10 ms，间隔远大于 60 ms
for start, end in gaps:
    pcm[start:end] = 0
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
    wf.writeframes(pcm.tobytes())

req = urllib.request.Request(
    "http://localhost:8000/v1/repair-gaps",
    data=json.dumps({
        "wav_base64": base64.b64encode(buf.getvalue()).decode(),
        "gaps": gaps,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req))
print(resp["repairs"])
# [{'start': 48000, 'end': 48240, 'method': 'predicted'},
#  {'start': 72000, 'end': 72480, 'method': 'predicted'}]
with open("repaired.wav", "wb") as f:
    f.write(base64.b64decode(resp["wav_base64"]))
EOF
```

```bash
# 生成 2 秒「前 0.5 s 纯噪声、之后 440 Hz 正弦 + 同源噪声」的 WAV，
# 以 [0, 24000) 为底噪参考段做全段降噪
python3 - <<'EOF'
import base64, io, json, wave, urllib.request
import numpy as np

rate = 48000
n = rate * 2
rng = np.random.default_rng(0)
t = np.arange(n) / rate
noise = 0.05 * rng.standard_normal(n)
sig = 0.4 * np.sin(2 * np.pi * 440 * t) + noise
sig[: rate // 2] = noise[: rate // 2]  # 前 0.5 s 只有底噪
pcm = np.round(sig * 32768).astype("<i2")
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
    wf.writeframes(pcm.tobytes())

req = urllib.request.Request(
    "http://localhost:8000/v1/denoise",
    data=json.dumps({
        "wav_base64": base64.b64encode(buf.getvalue()).decode(),
        "noise_interval": [0, 24000],   # 半开区间，0.5 s 纯噪声
        "strength": 0.8,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req))
with open("denoised.wav", "wb") as f:
    f.write(base64.b64decode(resp["wav_base64"]))
EOF
```

```bash
# 生成 2 秒「前 1 s 安静、后 1 s 响亮」的 WAV，做阈值 -20 dBFS、4:1 软拐点压缩
python3 - <<'EOF'
import base64, io, json, wave, urllib.request
import numpy as np

rate = 48000
t = np.arange(rate * 2) / rate
sig = 0.1 * np.sin(2 * np.pi * 440 * t)
sig[rate:] = 0.6 * np.sin(2 * np.pi * 440 * t[rate:])
pcm = np.round(sig * 32768).astype("<i2")
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
    wf.writeframes(pcm.tobytes())

req = urllib.request.Request(
    "http://localhost:8000/v1/compress",
    data=json.dumps({
        "wav_base64": base64.b64encode(buf.getvalue()).decode(),
        "threshold_dbfs": -20.0,
        "ratio": 4.0,
        "attack_ms": 5.0,
        "release_ms": 100.0,
        "knee_db": 6.0,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req))
with open("compressed.wav", "wb") as f:
    f.write(base64.b64decode(resp["wav_base64"]))
EOF
```

```bash
# 生成 3 秒 440 Hz 音频，以 speed=0.8 放慢到 3.75 秒（音高保持 440 Hz）
python3 - <<'EOF'
import base64, io, json, wave, urllib.request
import numpy as np

rate = 48000
t = np.arange(rate * 3) / rate
pcm = np.round(0.4 * np.sin(2 * np.pi * 440 * t) * 32768).astype("<i2")
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
    wf.writeframes(pcm.tobytes())

req = urllib.request.Request(
    "http://localhost:8000/v1/time-stretch",
    data=json.dumps({
        "wav_base64": base64.b64encode(buf.getvalue()).decode(),
        "speed": 0.8,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req))
print(resp["output_frames"])  # 180000 = round(144000 / 0.8)，即 3.75 s
with open("stretched.wav", "wb") as f:
    f.write(base64.b64decode(resp["wav_base64"]))
EOF
```

```bash
# 生成 2 秒双声道 WAV：右轨 = 左轨延迟 37 帧且反相，校准后右轨对齐并恢复极性
python3 - <<'EOF'
import base64, io, json, wave, urllib.request
import numpy as np

rate = 48000
n = rate * 2
rng = np.random.default_rng(0)
src = np.clip(np.round(rng.standard_normal(n) * 8000), -32768, 32767).astype(np.int64)
shift = 37
left = src
right = np.zeros(n, dtype=np.int64)
right[shift:] = -src[: n - shift]  # 右轨延迟 37 帧并反相
pcm = np.stack([left, right], axis=1).astype("<i2")
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(rate)
    wf.writeframes(pcm.tobytes())

req = urllib.request.Request(
    "http://localhost:8000/v1/align-stereo",
    data=json.dumps({
        "wav_base64": base64.b64encode(buf.getvalue()).decode(),
        "calibration_interval": [4800, 14400],   # 半开区间，0.2 s
    }).encode(),
    headers={"Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req))
print({k: v for k, v in resp.items() if k != "wav_base64"})
# => {'offset_samples': 37, 'polarity_inverted': True, 'correlation': -1.0}
with open("aligned.wav", "wb") as f:
    f.write(base64.b64decode(resp["wav_base64"]))
EOF
```

```bash
# 生成 2 秒「440 Hz + 8 kHz」双音 WAV，做 1 kHz +6 dB 与 8 kHz -6 dB 两段参量均衡
python3 - <<'EOF'
import base64, io, json, wave, urllib.request
import numpy as np

rate = 48000
t = np.arange(rate * 2) / rate
sig = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 8000 * t)
pcm = np.round(sig * 32768).astype("<i2")
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
    wf.writeframes(pcm.tobytes())

req = urllib.request.Request(
    "http://localhost:8000/v1/eq",
    data=json.dumps({
        "wav_base64": base64.b64encode(buf.getvalue()).decode(),
        "bands": [
            {"frequency_hz": 1000.0, "q": 1.0, "gain_db": 6.0},    # 先处理
            {"frequency_hz": 8000.0, "q": 2.0, "gain_db": -6.0},   # 后处理
        ],
        "ceiling_dbtp": -1.0,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req))
print({k: v for k, v in resp.items() if k != "wav_base64"})
# => {'input_true_peak_dbtp': ..., 'output_true_peak_dbtp': ..., 'post_gain_db': 0.0}
with open("eq.wav", "wb") as f:
    f.write(base64.b64decode(resp["wav_base64"]))
EOF
```

```bash
# 生成 2 秒「静音 + 三个脉冲咔哒声」的 WAV，检测瞬态起点用于剪辑定位
python3 - <<'EOF'
import base64, io, json, wave, urllib.request
import numpy as np

rate = 48000
pcm = np.zeros(rate * 2, dtype="<i2")
for pos in (24000, 48000, 72000):      # 0.5 s、1.0 s、1.5 s 处各一个咔哒声
    pcm[pos] = 26214
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
    wf.writeframes(pcm.tobytes())

req = urllib.request.Request(
    "http://localhost:8000/v1/onsets",
    data=json.dumps({
        "wav_base64": base64.b64encode(buf.getvalue()).decode(),
        "min_relative_flux": 0.1,       # 可省略，默认 0.1
    }).encode(),
    headers={"Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req))
print(resp["onsets"])
# [{'frame_index': 23808, 'strength': 0.92388},
#  {'frame_index': 47872, 'strength': 1.0},
#  {'frame_index': 71936, 'strength': 0.92388}]
EOF
```

```bash
# 生成 2 秒 440 Hz 原音与 1 秒指数衰减噪声脉冲响应，加 wet=0.3 的混响尾音
python3 - <<'EOF'
import base64, io, json, wave, urllib.request
import numpy as np

rate = 48000
t = np.arange(rate * 2) / rate
pcm = np.round(0.3 * np.sin(2 * np.pi * 440 * t) * 32768).astype("<i2")

rng = np.random.default_rng(0)
t_ir = np.arange(rate) / rate
ir = np.round(
    0.1 * np.exp(-t_ir / 0.15) * rng.standard_normal(rate) * 32768
).astype("<i2")

def enc(samples):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return base64.b64encode(buf.getvalue()).decode()

req = urllib.request.Request(
    "http://localhost:8000/v1/convolution-reverb",
    data=json.dumps({
        "wav_base64": enc(pcm),
        "ir_wav_base64": enc(ir),
        "wet": 0.3,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req))
# 输出 96000 + 48000 - 1 = 143999 帧，尾部即混响衰减
print(resp["output_frames"], resp["post_gain_db"])  # 143999 0.0
with open("reverb.wav", "wb") as f:
    f.write(base64.b64decode(resp["wav_base64"]))
EOF
```

## 项目结构

```
app/
  main.py          # FastAPI 入口与请求/响应模型；执行器生命周期与 503 映射
  executor.py      # 共享有界进程执行器：FIFO 队列、断开与异常的名额释放
  loudness.py      # WAV 校验、响度/真峰值测量、增益与量化
  compare_gain.py  # 增益/时移核验：归一化互相关、最小二乘增益、残差窗
  cut_splice.py    # 区间删除、保留片段拼接、接缝线性交叉淡化与标注映射
  repair_gaps.py   # 零值缺口修补：正则化 24 阶 LPC 双向外推融合/线性回退
  denoise.py       # 参考段谱减法降噪：STFT 噪声功率中位数估计与增益平滑
  compress.py      # 前馈压缩：声道联动包络、阈值/压缩比/启动释放/软拐点
  time_stretch.py  # WSOLA 变速不变调：40 ms 片段 / 20 ms 步长 / ±240 帧 NCC 搜索
  align_stereo.py  # 双声道对齐：校准区间去均值 NCC 偏移估计、右轨平移/反相
  eq.py            # 参量均衡：RBJ peaking biquad 串联、真峰值上限与后置增益回退
  onsets.py        # 谱通量瞬态检测：居中 Hann 窗 STFT、自适应阈值与 80 ms 选点
  convolution_reverb.py  # 卷积混响：FFT 线性卷积、干湿混合与全声道统一非正增益
tests/
  test_normalize.py
  test_compare_gain.py
  test_cut_splice.py
  test_repair_gaps.py
  test_denoise.py
  test_compress.py
  test_time_stretch.py
  test_align_stereo.py
  test_eq.py
  test_onsets.py
  test_convolution_reverb.py
  test_executor.py          # 执行器并发：满队列、两阶段断开、异常后容量恢复
  test_concurrency_http.py  # HTTP 层并发：503+Retry-After、/health 旁路、
                            # ASGI 取消与真实 socket 两阶段断开、容量恢复、生命周期
Dockerfile
docker-compose.yml
requirements.txt        # 运行依赖（已锁定版本）
requirements-dev.txt    # 测试依赖
```
