from my_vllm.entrypoints.llm import LLM

if __name__ == "__main__":
    llm = LLM(model="facebook/opt-125m")
    result = llm.generate("Hello, my name is", max_new_tokens=50)
    print(result)
