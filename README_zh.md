# ATR Framework

ATR Framework 是一个面向 ImgEdit 图像编辑任务的自动工具链框架。给定一张输入图像和一条编辑指令后，框架会先调用 Gemini 做图像理解和路由判断，再根据路由类别选择 Qwen agent 或 Banana agent 的工具链完成编辑，并保存中间结果、最终图像和 trace。

整体流程：

```text
input image + instruction
-> Gemini caption
-> Gemini router: A1 / A2 / B / C
-> agent planner selects tools
-> image editing / segmentation / crop-paste / verification
-> save caption.json, routing.json, trace.json, report.json and images
```

## 1. 环境准备

建议使用独立 conda 环境：

```bash
conda create -n atr python=3.10 -y
conda activate atr

cd ATR_Framework
pip install -r requirements.txt
```

如果服务器需要匹配 CUDA 版本的 PyTorch，请先按机器 CUDA 版本安装 PyTorch，再安装其余依赖。例如：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. API 配置

项目会调用 Gemini 做 caption、routing、fixprompt、ifinish，以及 Banana agent 的图像编辑。当前配置统一写在：

```text
configs/imgedit_pipeline.example.json
```

Google 认证支持两种方式，通过配置文件里的 `google.auth_mode` 选择。

### 2.1 Vertex AI service account

```json
"google": {
  "auth_mode": "vertex",
  "application_credentials": "/path/to/google_service_account.json",
  "project_id": "your-google-cloud-project-id",
  "location": "global",
  "api_key": ""
}
```

代码会把这些字段注入为环境变量，并用下面的方式初始化 client：

```python
genai.Client(vertexai=True, project=project_id, location=location)
```

### 2.2 Gemini API key

如果想用普通 Gemini API key，把配置改成：

```json
"google": {
  "auth_mode": "api_key",
  "application_credentials": "",
  "project_id": "",
  "location": "global",
  "api_key": "your-gemini-api-key"
}
```

代码会用下面的方式初始化 client：

```python
genai.Client(api_key=api_key)
```


## 3. 模型文件准备

### 3.1 Qwen-Image-Edit-2509

Qwen agent 会加载本地 Qwen-Image-Edit 模型。默认路径是：

```text
./examples/models/Qwen-Image-Edit-2509
```

也可以在配置文件里指定：

```json
"models": {
  "qwen_image_edit_path": "/path/to/Qwen-Image-Edit-2509"
}
```

下载链接：

```text
https://huggingface.co/Qwen/Qwen-Image-Edit-2509/tree/main
```

### 3.2 SAM3

B 类工具链会调用 `tools/sam3_tool.py`。当前代码需要两个路径：

```json
"models": {
  "sam3_dir": "/path/to/github/sam3/repo",
  "sam3_checkpoint": "/path/to/modelscope/sam3.pt"
}
```

含义：

```text
sam3_dir         SAM3 代码仓库根目录，目录下需要能看到 Python 包 sam3/
sam3_checkpoint  ModelScope 下载得到的 sam3.pt 权重文件
```

默认值是：

```text
./examples/sam3_repo
./examples/sam3_model/sam3.pt
```

SAM3 权重下载链接：

```text
https://www.modelscope.cn/models/facebook/sam3/files
```

常见目录形态：

```text
examples/
  sam3_repo/
    sam3/
    examples/
    scripts/
    pyproject.toml
  sam3_model/
    sam3.pt
    model.safetensors
    config.json
    tokenizer.json
```

## 4. 输入数据格式

单样本 JSON：

```json
{
  "index": "0",
  "input_image": "/path/to/image.jpg",
  "instruction": "Add a small brown dog sitting inside the open trunk of the antique car."
}
```

批量 JSONL：

```jsonl
{"index": "0", "input_image": "/path/to/image0.jpg", "instruction": "Add a small brown dog sitting inside the open trunk of the antique car."}
{"index": "1", "input_image": "/path/to/image1.jpg", "instruction": "Remove the pool table from the scene."}
```


### 4.1 ImgEdit Benchmark JSONL 准备

从 ImgEdit 数据集页面下载 `Benchmark.tar`：

```text
https://huggingface.co/datasets/sysuyy/ImgEdit/tree/main
```

把它放到任意本地目录后解压。解压后应得到一个本地 `Benchmark/` 目录：

```bash
tar -xf Benchmark.tar
```

生成 ATR 可直接读取的 JSONL：

```bash
python scripts/make_benchmark_jsonl.py \
  --benchmark-dir /path/to/Benchmark \
  --output-dir /path/to/output_jsonl
```

会生成：

```text
/path/to/output_jsonl/singleturn.jsonl
/path/to/output_jsonl/hard.jsonl
```

生成后的 `input_image` 字段会写成本地解压后的 `Benchmark/` 目录下对应图片的绝对路径。

## 5. 单样本运行

入口脚本：

```text
scripts/run_edit.py
```

Qwen agent：

```bash
python scripts/run_edit.py \
  --image /path/to/image.jpg \
  --instruction "Add a small brown dog sitting inside the open trunk of the antique car." \
  --agent qwen \
  --config configs/imgedit_pipeline.example.json \
  --output ./results
```

Banana agent：

```bash
python scripts/run_edit.py \
  --image /path/to/image.jpg \
  --instruction "Add a small brown dog sitting inside the open trunk of the antique car." \
  --agent banana \
  --config configs/imgedit_pipeline.example.json \
  --output ./results
```

使用 JSON 文件：

```bash
python scripts/run_edit.py \
  --json-file case.json \
  --agent qwen \
  --config configs/imgedit_pipeline.example.json \
  --output ./results
```

`run_edit.py` 每次运行一个样本；如果使用 Qwen agent，会在该进程中加载一次 Qwen-Image-Edit。

## 6. 批量运行

入口脚本：

```text
scripts/run_from_config.py
```

运行：

```bash
python scripts/run_from_config.py \
  --config configs/imgedit_pipeline.example.json
```

这个入口会按照配置文件启动固定 worker。Qwen agent 下，每个 worker 启动时加载一次 Qwen-Image-Edit，然后连续处理多个样本，直到 JSONL 全部跑完。

### 6.1 配置文件字段

当前示例配置：

```json
{
  "jsonl_file": "/path/to/your/input.jsonl",
  "output_dir": "/path/to/your/output/results",
  "agent": "qwen",
  "max_samples": null,
  "gpu_ids": [0],
  "max_workers": 1,
  "google": {
    "auth_mode": "vertex",
    "application_credentials": "/path/to/google_service_account.json",
    "project_id": "your-google-cloud-project-id",
    "location": "global",
    "api_key": ""
  },
  "models": {
    "gemini_model": "gemini-3-flash-preview",
    "qwen_image_edit_path": "/path/to/Qwen-Image-Edit-2509",
    "sam3_dir": "/path/to/github/sam3/repo",
    "sam3_checkpoint": "/path/to/modelscope/sam3.pt"
  }
}
```

字段说明：

```text
jsonl_file              输入 JSONL
output_dir              输出根目录
agent                   qwen / banana / both
max_samples             只跑前 N 条；null 表示全量
gpu_ids                 使用哪些 GPU；可以是 [0,1,2]，也可以是 "0,1,2"
max_workers             启动几个 worker
google                  Gemini / Vertex 认证配置
models.gemini_model     Gemini 模型名
models.qwen_image_edit_path  Qwen-Image-Edit 本地模型目录
models.sam3_dir         SAM3 代码仓库根目录
models.sam3_checkpoint  sam3.pt 权重文件
```


### 6.2 跳过已完成样本

`run_from_config.py` 支持按已有 `trace.json` 的 `status` 跳过样本。可在配置里加入：

```json
"skip_statuses": ["completed"]
```

这样如果输出目录里已经存在：

```text
{output_dir}/{agent}/{index}/trace.json
```

并且其中：

```json
{"status": "completed"}
```

该样本就不会重复跑。其他状态会重新跑并覆盖同名输出文件。

如果也想跳过强制完成的样本，可以写：

```json
"skip_statuses": ["completed", "completed_forced"]
```

## 7. 输出文件

默认输出结构：

```text
{output_dir}/{agent}/{index}/
```

其中 `{index}` 来自输入 JSON/JSONL 的 `index` 字段。常见输出：

```text
caption.json       Gemini 图像理解结果
routing.json       A1/A2/B/C 路由结果
trace.json         agent 工具调用轨迹
report.json        汇总报告
input_*.jpg        输入图像备份
process*.png       中间图像或最终图像
```

`trace.json` 里的 `index` 会和输入 JSON/JSONL 的 `index` 保持一致。

## 8. Agent 和工具链

### 8.1 Qwen agent

`agent=qwen` 使用本地 Qwen-Image-Edit 作为主要图像编辑后端：

```text
qwen_edit -> local QwenImageEditPlusPipeline
```

它需要本地 GPU 显存加载 Qwen-Image-Edit。

### 8.2 Banana agent

`agent=banana` 使用 Gemini 图像编辑作为主要编辑后端：

```text
qwen_edit -> gemini-2.5-flash-image
```

脚本中工具名仍叫 `qwen_edit`，但 Banana agent 的实现实际调用 Gemini image edit API。

### 8.3 路由类别

```text
A1: 直接编辑，通常直接调用图像编辑模型
A2: 需要先改写/增强编辑 prompt，再编辑
B: 需要 SAM3 分割目标，再编辑、粘贴或修复
C: 需要 crop 局部区域，编辑后再 paste 回原图
```

B/C 类会在正式 Step 1 前给 planner 一次 `fixprompt_tool` 机会。如果 planner 调用它，该调用会写入 trace 的 step0，但不会写进 planner history，也不会产生 process 图；系统会用改写后的 instruction 重新开始正式 Step 1。

## 9. 当前项目结构

```text
ATR_Framework/
  configs/
    imgedit_pipeline.example.json
  core/
    captioner.py
    router.py
    runtime_config.py
    agent_session_qwen.py
    agent_session_banana.py
  tools/
    crop_tool.py
    croppaste_tool.py
    fixprompt_tool.py
    ifinish_tool.py
    reprompt_tool.py
    sam3_tool.py
    smartpaste_tool.py
    target_tool.py
  prompts_qwen/
  prompts_banana/
  scripts/
    run_edit.py
    run_from_config.py
  ImgEdit/
    Basic/
    UGE/
  requirements.txt
  README.md
```






