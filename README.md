简短描述
- 一个基于 Qwen 2.5-7B 的简历文本分类微调项目。
- 本地部署Qwen 2.5-7B。
- 使用 bitsandbytes 的 4-bit 量化与 PEFT（LoRA）进行低成本微调，将分类任务表述为「在受限标签集合中生成单一标签」的生成式任务。
- 体现大模型微调、少量显存训练与工程化落地能力。
- 在受限显存环境下用 QLoRA + PEFT 成功微调大模型（Qwen 2.5-Instruct），显著降低训练资源消耗。
- 将分类问题设计为受约束的生成任务（prompt engineering）：模型直接输出标签字符串，训练与推理流程一致。
- 完整工程化流程：数据预处理、训练（SFTTrainer）、微调保存、PeftModel 加载推理、评估（classification_report + 混淆矩阵）。
- 实现了 tokenizer 特殊 token 管理与生成后处理，保证标签一致性与健壮性



 技术栈
- PyTorch, transformers, bitsandbytes, peft, trl（SFTTrainer）
- datasets（Hugging Face），pandas, scikit-learn, matplotlib, seaborn
- 量化：bitsandbytes nf4（4-bit）
