# !pip install llama-cpp-python

from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF",
	filename="EXAONE-4.0-1.2B-Q4_K_M.gguf",
)

llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "What is the capital of France?"
		}
	]
)