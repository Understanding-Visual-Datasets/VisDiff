# VisDiff API Server

## Design Choices

1. All LLMs/VLMs serve as API with cache enabled, because loading a LLM/VLM is expensive and we never modify them.
2. LLM functions in `utils_llm.py`, VLM functions in `utils_vlm.py`, others in `utils_general.py`.
3. Write unit tests to understand major functions.

## VLM Server Configuration

1. Conda install environments: `conda env create -f environment.yml`
2. Configure VLM
  - BLIP: no configuration needed
  - LLaVA: `git clone git@github.com:haotian-liu/LLaVA.git`
  - CogVLM: `git clone git@github.com:THUDM/CogVLM.git`; `wget https://huggingface.co/THUDM/CogVLM/resolve/main/cogvlm-chat.zip`; `unzip cogvlm-chat.zip`
3. Configure global variables in `global_vars.py`
4. Run `python vlm_server_[vlm].py`. It takes a while to load the VLM, especially the first time to download the VLM. (Note: concurrency is disabled as it surprisingly leads to worse GPU utilization)
5. Run `utils_vlm.py` to test the VLM.

## LLM Server Configuration

1. Pip install environments: `pip install vllm`
2. Configure global variables in `global_vars.py`
3. Run `python -m vllm.entrypoints.openai.api_server --model lmsys/vicuna-7b-v1.5`
4. Run `utils_llm.py` to test the LLM.

## Code Check-in Standards

1. Git create a new branch
2. Run the following before commit:
  - `pip install isort black`
  - `isort *.py; black *.py`
3. Create a pull request on Github
