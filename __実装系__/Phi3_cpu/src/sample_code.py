from llama_cpp import Llama
llm = Llama(
  model_path="(./models/Phi-3-mini-4k-instruct-q4.gguf",  # path to GGUF file
  n_gpu_layers=0, # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
)


prompt= [
    {"role": "system", "content": "あなたはAIアシスタントです。ユーザーの質問に対して適切な情報を提供してください。"},
    {"role": "user", "content": "あなたは何ができますか？網羅的かつ箇条書きで教えてください。"},
]


output = llm(
  f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
  max_tokens=512,
  stop=["<|end|>"],
  echo=True,
)


print(output['choices'][0]['text'])