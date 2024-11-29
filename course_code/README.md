# Setup.

1. Clone repository from https://github.com/Hersh245/crag
2. Follow Notion directions (CRAG Starter-Kit Code) to install dependencies

3. Create HuggingFace account, apply for access to `meta-llama/Llama-3.2-3B-Instruct`.
4. Get HuggingFace access token and log in:
```huggingface-cli login --token "your_access_token"```
5. Copy (or move) dataset into your directory:
```cp -r /home/hersh/crag/data /home/<your username>/crag/data```
6. Start VLLM server with following command. Note that the extra keyword is to ensure sequences fit on the GPU.
```vllm serve meta-llama/Llama-3.2-3B-Instruct --gpu_memory_utilization=0.85 --tensor_parallel_size=1 --dtype="half" --port=8088 --enforce_eager --max_model_len=45000```
7. Run `python generate.py`. Even if you run this from inside `course_code`, you need to specify dataset path as `data/x` or `example_data/x`, not `../data/x`.