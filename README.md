# ATR Framework

**Making Image Editing Easier via Adaptive Task Reformulation with Agentic Executions**
Paper: [arXiv:2604.15917](https://arxiv.org/abs/2604.15917)

ATR Framework is an automatic tool-chain framework for ImgEdit image editing tasks. Given an input image and an editing instruction, the framework first calls Gemini for image understanding and routing, then selects the Qwen agent or Banana agent tool chain according to the routing class. It saves intermediate results, the final image, and trace files.

Overall workflow:

```text
input image + instruction
-> Gemini caption
-> Gemini router: A1 / A2 / B / C
-> agent planner selects tools
-> image editing / segmentation / crop-paste / verification
-> save caption.json, routing.json, trace.json, report.json and images
```

## 1. Environment Setup

Using an isolated conda environment is recommended:

```bash
conda create -n atr python=3.10 -y
conda activate atr

cd ATR_Framework
pip install -r requirements.txt
```

If your server needs a PyTorch build that matches its CUDA version, install PyTorch first, then install the remaining dependencies. For example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. API Configuration

The project calls Gemini for captioning, routing, fixprompt, ifinish, and Banana agent image editing. The current unified configuration file is:

```text
configs/imgedit_pipeline.example.json
```

Google authentication supports two modes. Select one through `google.auth_mode` in the configuration file.

### 2.1 Vertex AI Service Account

```json
"google": {
  "auth_mode": "vertex",
  "application_credentials": "/path/to/google_service_account.json",
  "project_id": "your-google-cloud-project-id",
  "location": "global",
  "api_key": ""
}
```

The code injects these fields as environment variables and initializes the client as follows:

```python
genai.Client(vertexai=True, project=project_id, location=location)
```

### 2.2 Gemini API Key

To use a regular Gemini API key, change the configuration to:

```json
"google": {
  "auth_mode": "api_key",
  "application_credentials": "",
  "project_id": "",
  "location": "global",
  "api_key": "your-gemini-api-key"
}
```

The code initializes the client as follows:

```python
genai.Client(api_key=api_key)
```

## 3. Model Files

### 3.1 Qwen-Image-Edit-2509

The Qwen agent loads a local Qwen-Image-Edit model. The default path is:

```text
./examples/models/Qwen-Image-Edit-2509
```

You can also specify it in the configuration file:

```json
"models": {
  "qwen_image_edit_path": "/path/to/Qwen-Image-Edit-2509"
}
```

Download link:

```text
https://huggingface.co/Qwen/Qwen-Image-Edit-2509/tree/main
```

### 3.2 SAM3

The class-B tool chain calls `tools/sam3_tool.py`. The current code requires two paths:

```json
"models": {
  "sam3_dir": "/path/to/github/sam3/repo",
  "sam3_checkpoint": "/path/to/modelscope/sam3.pt"
}
```

Meaning:

```text
sam3_dir         Root directory of the SAM3 code repository. It must contain the Python package sam3/
sam3_checkpoint  sam3.pt checkpoint file downloaded from ModelScope
```

Default values:

```text
./examples/sam3_repo
./examples/sam3_model/sam3.pt
```

SAM3 checkpoint download link:

```text
https://www.modelscope.cn/models/facebook/sam3/files
```

Common directory layout:

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

## 4. Input Data Format

Single-sample JSON:

```json
{
  "index": "0",
  "input_image": "/path/to/image.jpg",
  "instruction": "Add a small brown dog sitting inside the open trunk of the antique car."
}
```

Batch JSONL:

```jsonl
{"index": "0", "input_image": "/path/to/image0.jpg", "instruction": "Add a small brown dog sitting inside the open trunk of the antique car."}
{"index": "1", "input_image": "/path/to/image1.jpg", "instruction": "Remove the pool table from the scene."}
```

### 4.1 Preparing ImgEdit Benchmark JSONL

Download `Benchmark.tar` from the ImgEdit dataset page:

```text
https://huggingface.co/datasets/sysuyy/ImgEdit/tree/main
```

Place it in any local directory and extract it. After extraction, you should have a local `Benchmark/` directory:

```bash
tar -xf Benchmark.tar
```

Generate ATR-compatible JSONL files:

```bash
python scripts/make_benchmark_jsonl.py \
  --benchmark-dir /path/to/Benchmark \
  --output-dir /path/to/output_jsonl
```

This creates:

```text
/path/to/output_jsonl/singleturn.jsonl
/path/to/output_jsonl/hard.jsonl
```

Each `input_image` field is written as the absolute path of the local image under the extracted `Benchmark/` directory.

## 5. Single-Sample Run

Entry script:

```text
scripts/run_edit.py
```

Qwen agent:

```bash
python scripts/run_edit.py \
  --image /path/to/image.jpg \
  --instruction "Add a small brown dog sitting inside the open trunk of the antique car." \
  --agent qwen \
  --config configs/imgedit_pipeline.example.json \
  --output ./results
```

Banana agent:

```bash
python scripts/run_edit.py \
  --image /path/to/image.jpg \
  --instruction "Add a small brown dog sitting inside the open trunk of the antique car." \
  --agent banana \
  --config configs/imgedit_pipeline.example.json \
  --output ./results
```

Using a JSON file:

```bash
python scripts/run_edit.py \
  --json-file case.json \
  --agent qwen \
  --config configs/imgedit_pipeline.example.json \
  --output ./results
```

`run_edit.py` processes one sample per run. If the Qwen agent is used, Qwen-Image-Edit is loaded once in that process.

## 6. Batch Run

Entry script:

```text
scripts/run_from_config.py
```

Run:

```bash
python scripts/run_from_config.py \
  --config configs/imgedit_pipeline.example.json
```

This entry starts fixed workers according to the configuration file. Under the Qwen agent, each worker loads Qwen-Image-Edit once at startup, then continuously processes multiple samples until the JSONL file is finished.

### 6.1 Configuration Fields

Current example configuration:

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

Field description:

```text
jsonl_file              Input JSONL
output_dir              Output root directory
agent                   qwen / banana / both
max_samples             Run only the first N samples; null means all samples
gpu_ids                 Which GPUs to use; can be [0,1,2] or "0,1,2"
max_workers             Number of workers to start
google                  Gemini / Vertex authentication configuration
models.gemini_model     Gemini model name
models.qwen_image_edit_path  Local Qwen-Image-Edit model directory
models.sam3_dir         Root directory of the SAM3 code repository
models.sam3_checkpoint  sam3.pt checkpoint file
```


### 6.2 Skipping Completed Samples

`run_from_config.py` can skip samples according to the `status` field in an existing `trace.json`. Add this to the configuration:

```json
"skip_statuses": ["completed"]
```

If the output directory already contains:

```text
{output_dir}/{agent}/{index}/trace.json
```

and the file contains:

```json
{"status": "completed"}
```

that sample will not be run again. Other statuses will be rerun and same-name output files will be overwritten.

To also skip forced-completed samples, use:

```json
"skip_statuses": ["completed", "completed_forced"]
```

## 7. Output Files

Default output structure:

```text
{output_dir}/{agent}/{index}/
```

`{index}` comes from the `index` field in the input JSON/JSONL. Common outputs:

```text
caption.json       Gemini image understanding result
routing.json       A1/A2/B/C routing result
trace.json         Agent tool-call trace
report.json        Summary report
input_*.jpg        Input image backup
process*.png       Intermediate or final edited images
```

The `index` in `trace.json` is kept consistent with the `index` in the input JSON/JSONL.

## 8. Agents and Tool Chains

### 8.1 Qwen Agent

`agent=qwen` uses local Qwen-Image-Edit as the main image editing backend:

```text
qwen_edit -> local QwenImageEditPlusPipeline
```

It needs local GPU memory to load Qwen-Image-Edit.

### 8.2 Banana Agent

`agent=banana` uses Gemini image editing as the main editing backend:

```text
qwen_edit -> gemini-2.5-flash-image
```

The tool name in the scripts is still `qwen_edit`, but the Banana agent implementation actually calls the Gemini image edit API.

### 8.3 Routing Classes

```text
A1: Direct editing, usually directly calling the image editing model
A2: Rewrite/enhance the editing prompt first, then edit
B: Use SAM3 to segment the target, then edit, paste, or refine
C: Crop a local region, edit it, then paste it back to the original image
```

For B/C classes, the planner gets one chance to call `fixprompt_tool` before formal Step 1. If the planner calls it, the call is written as step0 in the trace, but it is not added to planner history and does not produce a process image. The system then restarts formal Step 1 with the rewritten instruction.

## 9. Current Project Structure

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


