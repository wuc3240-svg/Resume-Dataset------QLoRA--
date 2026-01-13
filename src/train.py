import torch
from transformers import BitsAndBytesConfig
from src.config import *
from src.data_preprocessing import load_and_preprocess_data, split_dataset
from src.model_training import load_model_and_tokenizer, setup_lora_config, setup_training_args, train_model

def main():
    # 检查CUDA可用性
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # 数据预处理
    formatted_data, all_labels = load_and_preprocess_data(DATA_PATH)
    train_dataset, valid_dataset, test_dataset = split_dataset(
        formatted_data, 
        test_size=TRAIN_TEST_SPLIT, 
        validation_size=VALIDATION_SPLIT
    )
    
    # 4-bit量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16
    )
    
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, bnb_config)
    
    # 设置LoRA配置
    lora_config = setup_lora_config(
        r=LORA_R,
        alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        dropout=LORA_DROPOUT
    )
    
    # 设置训练参数
    training_args = setup_training_args(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=LOGGING_STEPS
    )
    
    # 训练模型
    trainer = train_model(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        lora_config=lora_config,
        tokenizer=tokenizer,
        training_args=training_args,
        max_seq_length=MAX_SEQ_LENGTH
    )
    
    # 保存最终模型
    trainer.save_model(FINAL_MODEL_DIR)
    print(f"模型已保存到: {FINAL_MODEL_DIR}")

if __name__ == "__main__":
    main()