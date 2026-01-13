import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def load_model_and_tokenizer(model_path, bnb_config):
    """加载预训练模型和tokenizer"""
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # 启用梯度检查点
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = "<|im_start|>"
    tokenizer.eos_token = "<|im_end|>"
    
    # 更新模型配置
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def setup_lora_config(r, alpha, target_modules, dropout):
    """设置LoRA配置"""
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return lora_config

def setup_training_args(output_dir, per_device_train_batch_size, gradient_accumulation_steps,
                        learning_rate, num_train_epochs, logging_steps):
    """设置训练参数"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        save_strategy="epoch",
        bf16=True,
        push_to_hub=False
    )
    return training_args

def train_model(model, train_dataset, valid_dataset, lora_config, tokenizer, training_args, max_seq_length):
    """训练模型"""
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    # 初始化训练器
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=lora_config,
        dataset_text_field="prompt",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args
    )
    
    # 开始训练
    trainer.train()
    
    return trainer