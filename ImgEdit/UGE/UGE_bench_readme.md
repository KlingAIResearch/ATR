
# Preprocessing
Before evaluating the model, you first need to use the provided JSON file (which contains metadata information) along with the original image files to generate the corresponding edited images by editing model. These edited images should be saved in a folder, with each image's filename prefix corresponding to the key value from the dictionary stored in the JSON file.

## Example Input/Output

The benchmark images can be downloaded from huggingface [Benchmark.tar](https://huggingface.co/datasets/sysuyy/ImgEdit/blob/main/Benchmark.tar)

### Input
A JSON file containing image edit instructions (`edit_json`):

```json
{
    "1": {"id": "000029784.jpg", "prompt": "Remove the lemon slice inside the glass, and turn the slice on the rim into an orange slice."},
    "2": {"id": "000065904.jpg", "prompt": "Remove the person in the front passenger seat."},
    ...
}
```


A folder containing original images (`origin_img_root`):

```folder
├── original_images                    
│   ├── 000029784.jpg                 
│   ├── 000065904.jpg                 
│   ...                
```

### Output:
A folder containing edited images, with filenames prefixed by the key value from the JSON file.

```folder
├── edited_images                    
│   ├── 1.png                 
│   ├── 2.png            
│   ...           
``` 

# Image Editing Evaluation using GPT

This project evaluates image editing processes using GPT-4o. The system processes a set of original and edited images, comparing them according to a predefined set of criteria, such as instruction adherence, image-editing quality, and detail preservation.

## Overview

The goal of this project is to evaluate the quality of image editing processes using GPT. The evaluation criteria include:
- **Instruction Adherence**: The edit must match the specified editing instructions.
- **Image-editing Quality**: The edit should appear seamless and natural.
- **Detail Preservation**: Regions not specified for editing should remain unchanged.


## Dependencies

The following Python libraries are required for running the script:
- `openai`: For interacting with the OpenAI API.

Install the required dependencies using `pip`:

```bash
pip install base64 tqdm tenacity openai
```

## Setup

1. **OpenAI API Key**: Make sure you have a valid OpenAI API key. Replace `"your api-key"` in the `call_gpt` function with your actual key.

2. **Images and JSON File**: You will need:
   - A folder containing the edited images (`--result_img_folder`).
   - A JSON file mapping keys to metadata and prompts for each image edit (`--edit_json`).
   - A root directory where the original images are stored (`--origin_img_root`).



## Usage

To run the script, use the following command:

```bash
python UGE_bench.py --result_img_folder <path_to_edited_images> --edit_json <path_to_edit_json> --origin_img_root <path_to_original_images> --num_processes <number_of_threads>
```

### Arguments:
- `--result_img_folder`: The directory containing the edited images.
- `--edit_json`: Path to the JSON file containing metadata and edit instructions.
- `--origin_img_root`: The root directory of the original images.
- `--num_processes`: The number of threads to use for processing. Default is 32.

### Example:

```bash
python UGE_bench.py --result_img_folder ./edited_images --edit_json ./UGE_edit.json --origin_img_root ./original_images --num_processes 4
```
## Example Input/Output

### Input:
A JSON file containing image edit instructions (`edit_json`):

```json
{
    "1": {"id": "000029784.jpg", "prompt": "Remove the lemon slice inside the glass, and turn the slice on the rim into an orange slice."},
    "2": {"id": "000065904.jpg", "prompt": "Remove the person in the front passenger seat."},
    ...
}
```

A folder containing original images (`origin_img_root`):

```folder
├── original_images                    
│   ├── 000029784.jpg                 
│   ├── 000065904.jpg                 
│   ...                 
```


A folder containing edited images(`result_img_folder`).

```folder
├── edited_images                    
│   ├── 1.png                 
│   ├── 2.png            
│   ...         
``` 

### Output:
A JSON file (`result_json`) with GPT evaluation for each image:

```json
{
    "1": "Brief reasoning: Instructions partially followed. Unnatural edit of people wearing shorts. Heavy alterations led to image distortion. Score: 2.",
    "2": "Brief reasoning: Ferrari logos replaced, but logo near text unchanged; edits slightly unnatural. Score: 3 (Acceptable)",
    ...
}
```


# Calculating the score

Calculate the average score for all edited images.

### Example

```bash
python get_average_score.py --result_json result.json
```

### Input
A JSON file (`result_json`) with GPT evaluation for each image:

```json
{
    "31": "Brief reasoning: Instructions partially followed. Unnatural edit of people wearing shorts. Heavy alterations led to image distortion. Score: 2.",
    "30": "Brief reasoning: Ferrari logos replaced, but logo near text unchanged; edits slightly unnatural. Score: 3 (Acceptable)",
    ...
}
```


### Output
A floating point number representing the average of all the scores, for example:

```
2.667
```

## Quick Start with Shell Script

You can run both evaluation steps with a single shell command.

### File Organization

```
your_project_folder/
├── UGE/
│   ├── run_uge_eval.sh         # The evaluation script
│   ├── UGE_bench.py
│   ├── get_average_score.py
│   ├── UGE_edit.json           # Edit metadata
│   └── uge_original_images/    # Original images directory
├── edited_images/              # Your edited images (can be in any location)
```

### Running the Evaluation

```bash
bash run_uge_eval.sh <path_to_edited_images>
```

### Example

```bash
bash run_uge_eval.sh ./edited_images
```

The script will automatically:
1. Run `UGE_bench.py` to evaluate edited images using GPT
2. Run `get_average_score.py` to calculate the final average score

### Output

The evaluation result will be printed to console showing:
- Total entries processed
- Valid scores extracted
- Success rate
- Final average score

### Notes

- Make sure you have a valid OpenAI API key configured
- The script uses 4 processes by default for parallel evaluation
- All paths in the shell script can be customized for your environment