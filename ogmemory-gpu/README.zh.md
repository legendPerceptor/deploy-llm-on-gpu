# oGMemory 多模态 GPU 实验部署

检查日期：2026-10-19。重启后已恢复 GPU/Docker 访问，并完成镜像构建、
模型下载和服务启动。实际验收范围及日志见
`../results/gpu-validation-2026-10-19-post-reboot/`；启动成功不代表完整基准通过。
本目录是独立 Compose 项目，不修改原有数据库或现有部署。

## 资源计划

重启后实测四张 Tesla T4（每张 15360 MiB），驱动 580.178.04，
Docker NVIDIA runtime 可用。GPU 0、2、3 通过容器 CUDA 矩阵运算；
GPU 1 通过 BGE-M3 实际向量推理。分配使用 `.env` 中的 GPU UUID。
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

## 文件与目标位置

当前生成目录：
`/home/yuanjian/Development/deploy-llm-on-gpu/ogmemory-gpu`

目标 oGMemory 仓库不在本会话可写范围。用户可在有权限的宿主机终端执行：

```bash
src=/home/yuanjian/Development/deploy-llm-on-gpu/ogmemory-gpu
dst=/home/yuanjian/Development/memory/oGMemory/deploy/gpu-models
# 目标已存在时停止，不覆盖已有配置。
test ! -e "$dst" && cp -r "$src" "$dst"
cd "$dst"
cp -n .env.example .env
chmod 600 .env
python3 -c 'import secrets; print(secrets.token_hex(32))'
# 将输出填入 .env 的 API_KEY，核对 OGMEMORY_DIR 和 GPU UUID。
```

也可以直接在当前目录管理模型，不必迁移。构建通过只读的额外 build context
复制 oGMemory 的 `scripts/cv_service.py`，不修改原仓库。

## 宿主机启动步骤

镜像与依赖不是“最新版本”的声明；本机验证范围以验收日志为准。
特别核对 Omni-3B 和 InsightFace 预训练权重的使用范围，不默认允许商业用途。

```bash
python3 preflight.py
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
  `cd ogmemory-gpu && docker compose up -d`（从工作区根目录执行）。
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
