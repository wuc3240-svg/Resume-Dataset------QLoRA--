import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def main():
    # 模型配置
    model_id = "../Qwen2.5-7B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # 加载模型和分词器
    print("正在加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        dtype=torch.float16,
        trust_remote_code=True
    )
    print("模型加载完成！")

    # 初始化对话历史
    conversation_history = []

    print("\n对话开始！输入'quit'或'exit'结束对话。")
    while True:
        # 获取用户输入
        user_input = input("\n你: ")
        if user_input.lower() in ['quit', 'exit']:
            print("对话结束！")
            break

        # 更新对话历史
        conversation_history.append(f"<|im_start|>user\n{user_input}<|im_end|>")
        conversation_history.append("<|im_start|>assistant")
        prompt = "".join(conversation_history)

        # 生成响应
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model.generate(**inputs, max_new_tokens=128, temperature=0.7, top_p=0.95)
        response = tokenizer.decode(out[0], skip_special_tokens=True)

        # 提取模型回复并更新对话历史
        assistant_response = response.split("assistant")[-1].strip()
        conversation_history[-1] = f"<|im_start|>assistant\n{assistant_response}<|im_end|>"

        # 打印回复
        print(f"\n助手: {assistant_response}")

if __name__ == "__main__":
    main()