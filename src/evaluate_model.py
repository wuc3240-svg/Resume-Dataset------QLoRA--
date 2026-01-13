import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 配置参数
MODEL_PATH = "../Qwen2.5-7B-Instruct"
DATA_PATH = "../data/ResumeDataSet.csv"
FINAL_MODEL_DIR = "../qwen-resume-classifier-final"
TEST_SIZE = 0.1

# 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16
)

def load_model():
    """加载微调后的模型"""
    config = PeftConfig.from_pretrained(FINAL_MODEL_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map="cuda"
    )
    model = PeftModel.from_pretrained(base_model, FINAL_MODEL_DIR)
    return model

def load_tokenizer():
    """加载tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_test_data():
    """加载并预处理测试数据"""
    data = pd.read_csv(DATA_PATH)
    data['Category'] = data['Category'].str.strip().str.lower()
    all_labels = sorted(data['Category'].unique())
    
    # 划分测试集
    test_data = data.sample(frac=TEST_SIZE, random_state=42)
    return test_data, all_labels

def classify_resume(model, tokenizer, resume_text, all_labels):
    """对单份简历进行分类"""
    # 将标签转换为小写以便匹配
    all_labels_lower = [label.lower() for label in all_labels]
    
    instruction = (
        f"请对下面的简历进行分类，仅可从以下标签中选择一个：{', '.join(all_labels)}。\n"
        "直接输出对应标签，无需解释。如果无法确定分类，请输出'other'。"
    )
    
    prompt = (
        f"<|im_start|>system\n{instruction}\n<|im_end|>\n"
        f"<|im_start|>user\n{resume_text}\n<|im_end|>"
        f"\n<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=50, 
        temperature=0.7,   
        do_sample=True,   
        top_p=0.9,         
        pad_token_id=tokenizer.eos_token_id  
    )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    

    if "<|im_start|>assistant" in decoded:
        decoded = decoded.split("<|im_start|>assistant")[-1]
    
    pred_clean = decoded.strip().lower()

    if pred_clean in all_labels_lower:
        pred_clean = all_labels[all_labels_lower.index(pred_clean)]
    else:
        matched = False
        for label in all_labels_lower:
            if label in pred_clean or pred_clean in label:
                pred_clean = all_labels[all_labels_lower.index(label)]
                matched = True
                break
        if not matched:
            pred_clean = "other"
    
    return pred_clean

def evaluate_model():
    """评估模型性能"""
    # 加载模型和tokenizer
    model = load_model()
    tokenizer = load_tokenizer()
    
    # 加载测试数据
    test_data, all_labels = load_test_data()
    
    # 进行预测
    predictions = []
    true_labels = []
    i = 0
    for _, row in test_data.iterrows():
        resume_text = row['Resume']
        true_label = row['Category']
        i += 1
        print("正在测试第",i,"个数据")
        pred = classify_resume(model, tokenizer, resume_text, all_labels)
        predictions.append(pred)
        true_labels.append(true_label)
    
    # 生成分类报告
    print("=== 模型分类报告 ===")
    print(classification_report(true_labels, predictions, labels=all_labels))
    
    # 生成混淆矩阵
    cm = confusion_matrix(true_labels, predictions, labels=all_labels)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(max(12, len(all_labels)*0.8), max(8, len(all_labels)*0.6)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=all_labels, yticklabels=all_labels, cbar=False)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    print("混淆矩阵已保存为 confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()