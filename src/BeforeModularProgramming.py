import numpy as np
import pandas as pd
import torch

print(torch.cuda.is_available())

data = pd.read_csv('../data/ResumeDataSet.csv')

# 对标签进行标准化处理（全部小写+去除首尾空白）
data['Category'] = data['Category'].str.strip().str.lower()

# 获取全部类别列表，用于prompt约束和评估映射
all_labels = sorted(data['Category'].unique())

# print(data.head())
# print(data.columns)
# print(f'行数：{len(data)}')

# from collections import Counter
# column_data = data['Category'].tolist()
# counter = Counter(column_data)
# for key, value in counter.items():
#     print(f'{key:<25}:{value:>3}')

#处理为可使用列表
def format_data(row):
    # 指令部分，带上标签约束
    instruction = (
        f"请对下面的简历进行分类，仅可选择标签：{all_labels}。"
        "直接输出对应标签，无需解释。"
    )
    input_text = row['Resume']
    output_text = row['Category']
    # 将三者拼成完整prompt，便于模型指令微调
    prompt = (
        f"<|im_start|>system\n{instruction}\n<|im_end|>\n"
        f"<|im_start|>user\n{input_text}\n<|im_end|>"
        f"\n<|im_start|>assistant\n"
    )
    return {
        "prompt": prompt,
        "output": output_text
    }

formatted_data = data.apply(format_data, axis=1)
formatted_data = pd.DataFrame(formatted_data.tolist())
# print(formatted_data.head())



from datasets import Dataset, DatasetDict

dataset = Dataset.from_pandas(formatted_data)
train_valid_test_dataset_dict = dataset.train_test_split(test_size = 0.2)
test_dataset = train_valid_test_dataset_dict['test']
train_valid_dataset = train_valid_test_dataset_dict['train']
train_valid_dataset_dict = train_valid_dataset.train_test_split(test_size = 0.2)
train_dataset = train_valid_dataset_dict['train']
valid_dataset = train_valid_dataset_dict['test']


# print(train_dataset.column_names)
# print(train_dataset.features)
# print(train_dataset.num_rows)
# print(train_dataset[:1])



from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16
)

# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "../Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
    low_cpu_mem_usage=True 
)



# 显卡太差
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# 显式设置tokenizer的特殊标记
tokenizer = AutoTokenizer.from_pretrained("../Qwen2.5-7B-Instruct", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token = "<|im_start|>"
tokenizer.eos_token = "<|im_end|>"

# 更新模型配置
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id


tokenizer = AutoTokenizer.from_pretrained("../Qwen2.5-7B-Instruct", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# QLoRA配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)






# 训练配置与启动
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="../qwen-resume-classifier",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True, 
    push_to_hub=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,  
    eval_dataset=valid_dataset,   
    peft_config=lora_config,
    dataset_text_field="prompt", 
    max_seq_length=256,
    tokenizer=tokenizer,
    args=training_args
)

trainer.train()
trainer.save_model("../qwen-resume-classifier-final")


import numpy as np
import pandas as pd



# 1. 模型推理
from peft import PeftModel, PeftConfig

# 加载微调后的模型
config = PeftConfig.from_pretrained("../qwen-resume-classifier-final")  # 路径加上../，与训练保存路径一致
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map="cuda"  # 修改为在cuda上与训练保持一致
)
model = PeftModel.from_pretrained(base_model, "../qwen-resume-classifier-final")  # 路径加上../

# 推理采用训练时的prompt格式，且使用test_dataset，all_labels、tokenizer都已在上方定义
def classify_resume(input_):
    # 判断如果是字典就取prompt否则直接用
    if isinstance(input_, dict) and "prompt" in input_:
        prompt = input_["prompt"]
    else:
        # 这里新简历拼prompt
        instruction = (
            f"请对下面的简历进行分类，仅可选择标签：{all_labels}。"
            "直接输出对应标签，无需解释。"
        )
        prompt = (
            f"<|im_start|>system\n{instruction}\n<|im_end|>\n"
            f"<|im_start|>user\n{input_}\n<|im_end|>"
            f"\n<|im_start|>assistant\n"
        )
    # 推理代码同上
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=20)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|im_start|>assistant" in decoded:
        decoded = decoded.split("<|im_start|>assistant")[-1]
    return decoded.strip()

# 2. 模型评估
from sklearn.metrics import classification_report

# 用test_dataset评估，列名已统一
predictions = []
true_labels = []

for sample in test_dataset:
    pred = classify_resume(sample["prompt"].split("<|im_start|>user\n")[1].split("\n<|im_end|>")[0])  # 提取简历原文
    # 由于模型可能输出多余空格或解释，标准化输出
    pred_clean = pred.strip().lower()
    if pred_clean not in all_labels:
        # 若模型输出非法标签，用'other'作为兜底
        pred_clean = "other"
    predictions.append(pred_clean)
    true_labels.append(sample["output"].strip().lower())

# 只对有效类别进行报告
print(classification_report(true_labels, predictions, labels=all_labels))

# 3. 结果可视化
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(true_labels, predictions, labels=all_labels)
plt.figure(figsize=(max(8, len(all_labels)), max(6, len(all_labels) * 0.6)))  # 动态调节大小
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=all_labels, yticklabels=all_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()  # 防止标签遮挡
plt.savefig("confusion_matrix.png")