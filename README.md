# oGMemory 多模态 GPU 实验部署

## 2026-10-23：完整视频对照状态

- 本机当前 `.env` 使用 `OMNI_MAX_IMAGES=16`；下文 7B 初始部署的 4 图限制是
  历史配置，`compose.omni7b.yaml` 在未设置变量时仍默认 4 图。
- Omni-7B 完成过 34 分 5 秒视频的 69 段记忆流水线（含失败片段恢复），
  但答案质量未通过验收；30B 完成部署可行性、少量多模态诊断和基于旧 7B
  记忆的 QA 对照，**没有完成 30B 全视频记忆重建**。
- 同输入 MiniMax 对照最终完成 69 条记忆；QA 为 11/12 有效答案，末题在
  2048 和独立 8192 输出预算试验中均耗尽 reasoning 预算。
  两种评委均认为其答案优于 7B，但单样本自动评分不等于可信准确率，
  也不能将该运行报告为全流程成功。
- 详细报告在相邻 oGMemory 仓库的
  `docs/m3_sample1_omni7b_20261022.md`、
  `docs/m3_minimax_comparison_20261023.md`；30B 部署说明见
  `QWEN3_OMNI_TRIAL.md`。本仓库的原始结果保留在忽略的 `results/`，
  凭据、私有配置、模型权重及运行产物不提交。

## 2026-10-22：双卡 Omni-7B 本地验证

新增可选覆盖文件 `compose.omni7b.yaml`，保留下面的 3B 基础配置：

- Omni-7B：GPU **0、3**，FP16、TP=2、32768 上下文、单并发、最多 4 张图。
- 固定模型 revision：`ae9e1690543ffd5c0221dc27f79834d0294cba00`；
  五个 safetensors 分片已逐个校验 SHA-256。
- embedding 仍在 GPU 1，人脸服务仍在 GPU 2，不搬迁或清空数据库/缓存卷。
- 7B 使用 `llama3_json` 解析器读取实际生成的 JSON 函数调用；
  专用模板为 `templates/qwen2_5_omni7b_json_tools.jinja`。
  3B 原来的 Hermes 配置及模板保持不变。已验证 7B 的非流式自动调用和
  工具结果回传，不能据此保证复杂工具选择或所有流式调用。
- 原 Hermes 配置下，7B 会输出裸 JSON 而不是结构化工具调用。
  本机 vLLM 0.10.2 的 `required` 工具输出仍可能出现数组解析 400；
  oGMemory 对明确的输出协议错误使用已有的严格 JSON 抽取回退。
  普通文本或 API 健康检查不算工具调用成功。

本机忽略的 `.env` 已加入
`COMPOSE_FILE=compose.yaml:compose.omni7b.yaml:compose.text7b.yaml`，因此默认
`docker compose up` 会保留 Omni-7B 和下面的文本侧车。也可以显式启动 Omni：

```bash
docker compose -f compose.yaml -f compose.omni7b.yaml up -d --no-deps --no-build omni
set -a; source .env; set +a
python3 scripts/smoke_test.py --model omni-7b --tools \
  --image results/gpu-validation-2026-10-19-post-reboot/astronaut.jpg \
  --audio results/gpu-validation-2026-10-19-post-reboot/tone.wav
```

回滚模型时显式使用 `docker compose -f compose.yaml up -d --no-deps --no-build omni`，
并同步应用的模型名、密钥和真实上下文；不能只回滚容器后仍让应用请求 `omni-7b`。
不要删除命名卷。运行前的私有环境备份在本次结果目录的 `private-backups/`，
不要公开该目录。

本次完整运行日志、权重校验、GPU 采样及验收报告存放在
`results/local-e2e-20261022/`。M3 复现说明见 oGMemory 的
`docs/m3_local_gpu_validation.md`。下面的 10 月 19–20 日内容为历史 3B 验收记录，
不代表当前运行模型，也不代表完整基准结果。

### 本地文本侧车与 LoCoMo

Omni-only 的 LoCoMo 长会话抽取在 2048/4096 输出预算下均出现截断，
相关尝试仅保留为失败诊断，不作为完成的基准结果。
最终文本侧车 `compose.text7b.yaml` 使用
`Qwen/Qwen2.5-7B-Instruct-AWQ`，固定 revision
`b25037543e9394b818fdfca67ab2a00ecc7dd641`，两个权重分片已校验 SHA-256。
它在 GPU 1 与 embedding 共用，端口 `8002`、模型名 `text-7b`、
32768 上下文、单并发、显存利用率 0.82。实际文本/工具闭环/embedding 联合冒烟通过。

T4 上启用 chunked prefill 曾导致共享内存内核错误（需要 98304 bytes，
硬件限制 65536），因此该配置关闭了这项优化。0.70 显存利用率不足以同时容纳
全长 prefill 峰值和 32768 KV cache，最终使用 0.82。这是已测小并发配置，
不是任意长输入/并发的保证。重新跑 M3 音频时使用 GPU 2，或先停文本侧车，
不要再把音频 worker 与它同时放在 GPU 1。

本机隔离的最终 LoCoMo 服务定义在 `compose.evaluation.yaml`：

- oGMemory：`127.0.0.1:4833`；网关：`127.0.0.1:48545`。
- 导入数据库：`ogmem_local_text7b_20261022`；最终冻结 QA 数据库：
  `ogmem_local_text7b_frozen_20261022`。原数据库不清空，失败尝试保存在另行命名的实验数据库中。
- 账号 `local-locomo-text7b-20261022`，独立网关状态目录；
  抽取和回答用 `text-7b`，judge 用本地 `omni-7b`，embedding 用本地 BGE-M3。
- 配置/凭证引用本次结果目录下的私有 env 文件和 oGMemory 忽略的 YAML，
  不适合作为无凭证的新机器通用启动配置。
- 当前网关关闭自动捕获、保留自动检索，使用新的 QA 会话目录。
  QA 前后记忆节点、向量和归档的哈希保持一致。不要用这个冻结配置重新导入数据。

```bash
docker compose -f compose.evaluation.yaml config --quiet
docker compose -f compose.evaluation.yaml up -d
cd ../memory/locomo-test
.venv/bin/python -m locomo_test.cli run configs/ogmem-local-text7b-frozen.toml --resume
```

`--resume` 跳过导入，仅适用于已经确认导入完成的本次结果目录。
LoCoMo small 原文件有 50 题，现有 runner 排除 category 5 后实际评测 35 题。
初轮开启捕获的 QA 会把评测问答写回记忆，其结果已排除。最终冻结轮完成 35/35 QA，
但两个本地评委仍有明确的日期/时间推理误判；原始分数不是可信准确率。
最终结论、判分分歧和快照哈希见 `results/local-e2e-20261022/VALIDATION_REPORT.md`。

检查日期：2026-10-19。重启后已恢复 GPU/Docker 访问，并完成镜像构建、
模型下载和服务启动。实际验收范围及日志见
`results/gpu-validation-2026-10-19-post-reboot/`；启动成功不代表完整基准通过。
本目录是独立 Compose 项目，不修改原有数据库或现有部署。

## 资源计划

重启后实测四张 Tesla T4（每张 15360 MiB），驱动 580.178.04，
Docker NVIDIA runtime 可用。GPU 0、2、3 通过容器 CUDA 矩阵运算；
GPU 1 通过 BGE-M3 实际向量推理。分配使用 `.env` 中的 GPU 索引或 UUID，推荐 UUID。
原准备阶段的设备和 Docker 权限阻塞已解除。

| 物理 GPU | 第一阶段任务 | 对外接口 | 策略 |
|---|---|---|---|
| 0 | Qwen2.5-Omni-3B：图像、音频、文本理解 | `127.0.0.1:8000/v1` | FP16，单并发，显存利用率参数 0.88 |
| 1 | BGE-M3：文本向量 | `127.0.0.1:8001/v1` | 1024 维；不是原始图像/音频向量 |
| 2 | InsightFace buffalo_l：人脸检测及身份向量 | `127.0.0.1:18081` | 512 维，要求 ONNX 会话首选 CUDA |
| 3 | 预留 ASR、说话人分离和声纹提取 | 暂无容器 | Whisper → pyannote → ECAPA 串行；后续集成 |

这里只实现前三项的部署定义。GPU 3 **仅预留**，`.env` 中 AUDIO_GPU 不会启动服务。
四张卡不是一张统一大显存卡；Compose 设备选择也不是显存配额。
先用 `nvidia-smi` 核对 UUID、占用、驱动，不抢占已有任务。

第二阶段可以先停掉第一阶段服务，再测试 Omni-7B 占 GPU 0–1、
embedding 在 GPU 2、音频在 GPU 3；人脸结果提前预计算。
该阶段尚未提供可直接启动的 Compose 配置，不要同时运行两种分配。

## 唯一部署入口：工作区根目录

所有命令均从 `/home/yuanjian/Development/deploy-llm-on-gpu` 执行。
原内层 `ogmemory-gpu/` 已合并到根目录，不再维护第二套 Compose 或环境配置。

```text
compose.yaml          # 唯一 Compose 入口：omni / embeddings / cv-gpu
.env                  # 本机密钥和参数，不提交 Git
.env.example          # 新环境配置模板
Dockerfile.omni       # vLLM + 音频依赖
Dockerfile.cv         # InsightFace GPU 镜像
requirements-cv.txt
.dockerignore         # 排除密钥、Git、日志及模型验证样本
scripts/              # preflight.py / preflight.sh / cv_gpu.py / smoke_test.py
tests/
results/              # 历史验收记录，不作为当前状态声明
```

整理时保留了外层 `.env` 的已有值（包括 API_KEY、HF_TOKEN、上下文和 revision），
只从内层补齐缺失配置。迁移前两份环境文件备份在
`results/layout-migration-2026-10-19/`（目录权限 700、文件权限 600）。
本机旧 `.env` 中未使用的 `OMNI_7B_*`、`VIDEO_LIMIT` 不会启动额外服务。
旧外层可选 7B 比较配置不再作为部署入口，避免与人脸服务抢占 GPU 2。

项目名保留为 `t4-omni`，以管理当前外层已启动的同名容器；三个模型卷显式命名为
`ogmemory-gpu_hf-cache`、`ogmemory-gpu_embedding-cache`、`ogmemory-gpu_face-cache`，
复用此前验收时已下载的模型。旧 `t4-omni_*` 缓存卷不删除。
`docker compose up -d` 才会将现有容器切换到整理后的配置；移动文件本身不会重启服务。
不要再从旧路径执行命令，也不要同时启动旧项目。

新机器首次准备（不要覆盖已有 `.env`）：

```bash
cp -n .env.example .env
chmod 600 .env
python3 -c 'import secrets; print(secrets.token_hex(32))'
# 将生成值填入 API_KEY；核对 OGMEMORY_DIR、GPU、端口及模型 revision。
```

CV 构建通过只读的额外 build context 复制 oGMemory 的 `scripts/cv_service.py`，
不修改原仓库或数据库。

### 调用本地模型

`/v1` 是基础地址，聊天请求必须发送到 `/v1/chat/completions`；
对外模型名为 `omni-3b`，不是模型仓库名称。

```bash
export API_KEY="$(sed -n 's/^API_KEY=//p' .env)"
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{"model":"omni-3b","messages":[{"role":"user","content":"你好"}],"max_tokens":128}'
python3 scripts/smoke_test.py
```

### 工具调用（tool calling）

Omni 服务显式启用 `--enable-auto-tool-choice --tool-call-parser hermes`，
并通过只读挂载加载 `templates/qwen2_5_omni_tools.jinja`。
本机缓存的 Omni-3B 原始模板不处理 `tools` / 历史 `tool_calls`，因此不能只加
解析器参数。自定义模板保留 Omni 图像、音频和视频占位符，同时加入工具定义、
`<tool_call>` 和 `<tool_response>` 格式；不要直接换用纯文本模型模板。

应用配置（会短暂中断 Omni，不重建镜像或其他服务）与验收：

```bash
docker compose up -d --no-deps --no-build omni
export API_KEY="$(sed -n 's/^API_KEY=//p' .env)"
python3 scripts/smoke_test.py --tools
```

客户端在 `/v1/chat/completions` 中提交 `tools` 和 `tool_choice: "auto"`。
返回 `message.tool_calls` 后由**客户端**执行工具，并将 assistant 消息及
`role: "tool"`、匹配的 `tool_call_id`、字符串形式的 `content` 回传；
模型服务不会执行工具。工具须由应用做授权与参数校验。

`--tools` 检查非流式自动调用的函数名、JSON 参数、调用 ID 和工具结果回传后的回答，
不把普通文本中的工具标签当作成功。该测试不代表复杂工具选择、多工具并行、
流式调用或完整 QA 基准已通过，也不保证 3B 模型的工具选择质量。
本地模板单测需要 `jinja2`（服务镜像已包含，本机测试环境也需安装）。

2026-10-20 本机验收：运行中的 vLLM 0.10.2 / Omni-3B 已应用以上配置，
`--tools` 非流式调用及工具结果回传通过；现有 `astronaut.jpg`、`tone.wav`
样本的单图、单音频、图像＋音频请求，以及文本和 BGE-M3 embeddings 回归通过。
13 项本地单测、Compose 配置检查通过。这是小负载冒烟验收，不是完整流水线基准。

预检要求随机 API_KEY 至少 32 字符。如果现有密钥较短，预检会报错，
但目录整理不会擅自替换密钥；请自行更改 `.env` 并同步客户端后再重建服务。

## 宿主机启动步骤

镜像与依赖不是“最新版本”的声明；本机验证范围以验收日志为准。
特别核对 Omni-3B 和 InsightFace 预训练权重的使用范围，不默认允许商业用途。

```bash
python3 scripts/preflight.py
# 此检查只读配置、Docker 与 GPU 清单；不会创建容器或修改驱动。

docker compose pull embeddings
docker compose build omni cv-gpu

# 先验证每张卡上的实际 CUDA，而不只是镜像能拉取。
docker compose run --rm --no-deps --entrypoint python3 omni \
  -c 'import torch; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda"))'
docker compose run --rm --no-deps --entrypoint python cv-gpu \
  -c 'import torch; import onnxruntime as o; print(torch.ones(1,device="cuda")); print(o.get_available_providers())'

# 分阶段启动，便于定位显存/模型兼容性问题。
docker compose up -d omni embeddings
docker compose logs --tail=100 -f omni embeddings
# 两个端点就绪后再启动人脸服务。
docker compose up -d cv-gpu
docker compose ps
curl --fail http://127.0.0.1:18081/health/gpu
```

实际模型下载可能较慢。CV 启动器会校验检测和识别模型的真实 ONNX session
provider，发现 CPU 回退就失败；它不保证每个 ONNX 节点均在 GPU 上执行。
物体检测和姿态接口明确返回 501，避免把原占位接口的空列表当作实验结果。
人脸服务无认证，仅绑定宿主机 loopback，不要改为公网监听。
模型端点带 API key，也仅绑定 loopback。命名卷保存下载缓存。

TEI 未假定内置 curl/Python，因此未配置依赖这些工具的容器健康检查；
必须通过 `/v1/embeddings` 的实际请求验证就绪。

### 本机部署修复与日常操作

- Omni 从 `VLLM_IMAGE` 指定的基础镜像构建为 `ogmemory-omni:local`，
  补充 `librosa==0.11.0`，否则音频输入会返回缺少 `vllm[audio]` 的错误。
- CV 使用 `onnxruntime-gpu==1.20.2`；重装 `ninja==1.11.1.3` 修复基础镜像的
  包平台元数据，并通过 `ldconfig` 注册 NVIDIA Python 包的动态库目录。
  仅导入 torch 不足以保证 ONNX Runtime 找到全部 cuDNN 子库。
- 首次下载曾发生 BGE 权重下载中断、Omni 第一分片损坏。
  已补齐缓存、重新下载损坏分片；Omni 三个分片均通过 SHA-256 校验。
  Omni 禁用 Xet 下载路径。以后若下载中断，应校验文件，不能仅凭文件大小认定完整。
- 保留原 `restart: 'no'` 策略，**机器重启后不会自动恢复服务**。手动启动：
  `docker compose up -d`（从工作区根目录执行）。
- 查看状态：`docker compose ps`。健康检查：
  `curl --fail http://127.0.0.1:18081/health/gpu`。
- 模型 API 的密钥保存在本目录权限为 600 的 `.env` 中。
  上游启动日志可能输出 API key，不要公开原始日志或 `docker compose config` 输出。

## oGMemory 接入与实验边界

基于检查过的 `config/m3/pipelines/video_episodic_full.yaml`、
`scripts/cv_service.py` 和现有部署方案：

- 顶层 `llm` 以及 `m3.episodic_llm` 使用 OpenAI provider、
  `http://127.0.0.1:8000/v1`、模型名 `omni-3b` 和新 API_KEY。
- `m3.qa_llm` 与 `m3.judge_llm` 是独立覆盖项，须显式核对，避免意外请求外部服务。
  QA 的 tool calling、JSON 输出仍需验证，不能由 `/v1/models` 成功推断。
- 顶层 `embedding` 和 `m3.embedding` 指向
  `http://127.0.0.1:8001/v1`，模型 `BAAI/bge-m3`；相关向量维度设为 1024。
- 设置 `OGMEM_CV_SERVICE_URL=http://127.0.0.1:18081`。
- 保留现有数据库连接，使用新的实验向量表/集合和运行目录，重新建立索引。
  即便维度相同，也不要混用旧模型向量。
- 上述 localhost 地址适用于宿主机进程或现有 host-network 容器；
  普通 bridge 容器的 localhost 不是宿主机，需另行配置网络和可达地址。

不要覆盖生产 `ogmemory.yaml`，复制一份实验配置后按上述实际字段调整。

### 小负载与完整负载必须区分

本配置默认 4 帧、8192 总上下文、单请求，只是待测的小负载候选，不保证能放进 T4。
应用的抽帧和 episodic 阶段都要降到最多 4 帧，输出 token 预算先降到 512。
音频也先用短片段。服务端的帧上限不会自动截断应用请求。

完整流水线默认最多 16 帧、30 秒音频。只有单图、单音频、混合输入均通过后，
才将 `OMNI_MAX_IMAGES=16`，恢复应用两处帧配置，并用真实输入测上下文和显存峰值。
不要把小负载结果标为完整基准。遇到上下文不足/OOM，减小输入或调整分阶段方案，
不要直接增加并发；总上下文必须容纳输入、多模态 token 和输出预算。

音频阶段后续需要显式设置 `voice_asr.params.device`、
`speaker_diarization.params.device`、`speaker_embedding.params.device` 为 `cuda`，
将处理进程限制在预留 GPU 上，并补齐其运行环境。pyannote 需授权模型访问。
当前 Compose **不提供音频 worker**，也不能保证宿主机 Python 已具备这些依赖。

## 验收与回滚

1. 导出与 `.env` 相同的 API_KEY，运行原工作区的端点测试：
   `python3 /home/yuanjian/Development/deploy-llm-on-gpu/scripts/smoke_test.py --image /path/sample.jpg --audio /path/sample.wav`。
2. `curl --fail -H 'Content-Type: application/octet-stream' --data-binary @/path/face.jpg http://127.0.0.1:18081/detect/faces`；
   用确实含脸且允许使用的图片检查非空检测和有限的 512 维向量。
3. 用隔离数据跑一个视频，确认真实 ASR/声纹、episodic schema、索引、检索与 QA，
   不把 skipped 或退化算法当成成功。
4. 再用固定 10 视频测质量、延迟、GPU 峰值、OOM、失败率，之后才增加并发。
5. 记录镜像 digest、模型 commit、应用代码版本、抽帧参数和独立运行目录。
   当前 `main` 模型 revision 是可变的，正式实验前固定 commit。

停止模型：`docker compose down`。不要加 `-v`，不要 prune，不操作其他 Compose
项目。此操作不删除命名缓存卷，也不触碰应用数据库。

## 本地结构与配置检查

```bash
docker compose --env-file .env.example config --quiet
python3 -m unittest discover -s tests -v
bash -n scripts/preflight.sh
```

上述检查不启动容器、不下载模型；当前运行状态以 `docker compose ps` 和实际请求为准。
