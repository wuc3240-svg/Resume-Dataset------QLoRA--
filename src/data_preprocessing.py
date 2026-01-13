import pandas as pd
from datasets import Dataset, DatasetDict

def load_and_preprocess_data(data_path):
    """加载并预处理简历数据"""
    # 加载数据
    data = pd.read_csv(data_path)
    
    # 标签标准化
    data['Category'] = data['Category'].str.strip().str.lower()
    all_labels = sorted(data['Category'].unique())
    
    # 数据格式化
    def format_data(row):
        instruction = (
            f"请对下面的简历进行分类，仅可选择标签：{all_labels}。"
            "直接输出对应标签，无需解释。"
        )
        input_text = row['Resume']
        output_text = row['Category']
        
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
    
    return formatted_data, all_labels

def split_dataset(formatted_data, test_size=0.2, validation_size=0.2):
    """划分训练集、验证集和测试集"""
    dataset = Dataset.from_pandas(formatted_data)
    
    # 划分训练集和测试集
    train_valid_test = dataset.train_test_split(test_size=test_size)
    train_valid = train_valid_test['train'].train_test_split(test_size=validation_size)
    
    train_dataset = train_valid['train']
    valid_dataset = train_valid['test']
    test_dataset = train_valid_test['test']
    
    return train_dataset, valid_dataset, test_dataset