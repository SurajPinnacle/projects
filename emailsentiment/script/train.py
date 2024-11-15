import argparse
import logging
import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoConfig,
    DataCollatorWithPadding,
)
from transformers.trainer_utils import EvalPrediction
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer model on custom data.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Pretrained model name or path")
    parser.add_argument("--train_file", type=str, help="Path to the training data CSV")
    parser.add_argument("--validation_file", type=str, help="Path to the validation data CSV")
    parser.add_argument("--output_dir", type=str, help="Output directory for model")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max input sequence length")
    parser.add_argument("--do_train", action="store_true", help="Whether to train the model")
    parser.add_argument("--do_eval", action="store_true", help="Whether to evaluate the model")
    parser.add_argument("--pad_to_max_length", action="store_true", help="Pad sequences to max length")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load dataset
    data_files = {}
    if args.train_file:
        data_files["train"] = args.train_file
    if args.validation_file:
        data_files["validation"] = args.validation_file
    raw_datasets = load_dataset("csv", data_files=data_files)

    # Get unique labels and determine weights for imbalanced classes
    label_list = raw_datasets['train'].unique('label')
    label_list.sort()
    num_labels = len(label_list)
    
    label_map = {label: i for i, label in enumerate(label_list)}
    raw_datasets = raw_datasets.map(lambda examples: {'label': label_map[examples['label']]})
    
    # Calculate class weights
    labels = raw_datasets['train']['label']
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    # Adjust model to handle class weights
    model.config.class_weights = class_weights

    # Tokenize data
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding="max_length" if args.pad_to_max_length else False, truncation=True, max_length=args.max_seq_length)

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    # Custom metrics for evaluation
    def compute_metrics(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average=None, labels=[label_map['Negative']])
        return {
            "precision_negative": precision[0],
            "recall_negative": recall[0],
            "f1_negative": f1[0]
        }

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy="epoch" if args.do_eval else "no",
        save_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
    )

    # Define Trainer with weighted loss
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=model.config.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # Initialize WeightedTrainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if args.do_train:
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # Evaluation
    if args.do_eval:
        eval_result = trainer.evaluate()
        logger.info(f"Evaluation result: {eval_result}")

if __name__ == "__main__":
    main()
