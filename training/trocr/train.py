from dotenv import dotenv_values
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from training.trocr.dataset import FoodDataset
from transformers import default_data_collator
import wandb
import torch
import numpy as np
from training.trocr.config import MODEL_CONFIG, DATALOADER_CONFIG, TRAIN_CONFIG, EVAL_CONFIG, CALLBACK_CONFIG

env_values = dotenv_values(".env")
os.environ["WANDB_API_KEY"] = env_values["WANDB_API_KEY"]

wandb.init(project=env_values["PROJECT_NAME"])

checkpoint = MODEL_CONFIG["checkpoint"]

processor = TrOCRProcessor.from_pretrained(checkpoint)

model = VisionEncoderDecoderModel.from_pretrained(checkpoint)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
# TODO: check if this works
model.config.eos_token_id = processor.tokenizer.eos_token_id

train_dataset = FoodDataset(split="train", processor=processor, max_target_length=DATALOADER_CONFIG["max_length"])
val_dataset = FoodDataset(split="val", processor=processor, max_target_length=DATALOADER_CONFIG["max_length"])
val_dataset = torch.utils.data.Subset(val_dataset, indices=range(256))


### LOG .PY FILES
with open("training/trocr/config.py", "r") as config_file:
    config_content = config_file.read()
    wandb.log({"config_file_content": wandb.Html(f"<pre>{config_content}</pre>")})
    
with open("training/trocr/train.py", "r") as train_file:
    train_content = train_file.read()
    wandb.log({"train_file_content": wandb.Html(f"<pre>{train_content}</pre>")})
    
with open("training/trocr/dataset.py", "r") as dataset_file:
    dataset_content = dataset_file.read()
    wandb.log({"dataset_file_content": wandb.Html(f"<pre>{dataset_content}</pre>")})
    
### DEFINE METRICS

def extract_value_from_text(text: str, key: str) -> float:
    try:
        return float(text.split(f"{key} per 100g: ")[1].split("\n")[0])
    except:
        return -100

def extract_values_from_text(text: str) -> tuple:
    keys = ["kcal", "proteins", "carbs", "fats"]
    return tuple(extract_value_from_text(text, key) for key in keys)

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_logits = pred.predictions[0]
    pred_ids = np.argmax(pred_logits, axis=-1)

    decoded_preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    decoded_labels = processor.batch_decode(labels_ids, skip_special_tokens=True)
    
    # log first 20 predictions and labels
    wandb.log(
        {
            "predictions": wandb.Table(
                columns=["prediction", "label"],
                data=[[decoded_preds[i], decoded_labels[i]] for i in range(20)],
            )
        }
    )

    preds_values = [extract_values_from_text(pred) for pred in decoded_preds]
    labels_values = [extract_values_from_text(label) for label in decoded_labels]

    mae_values = np.mean(np.abs(np.array(preds_values) - np.array(labels_values)), axis=0)
    accuracies = np.mean(np.array(preds_values) == np.array(labels_values), axis=0)
    
    return {
        **{f"{key}_mae": value for key, value in zip(["kcal", "proteins", "carbs", "fats"], mae_values)},
        **{f"{key}_accuracy": value for key, value in zip(["kcal", "proteins", "carbs", "fats"], accuracies)}
    }
### TRAIN

train_args = Seq2SeqTrainingArguments(
    
    per_device_train_batch_size=DATALOADER_CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=DATALOADER_CONFIG["per_device_eval_batch_size"],
    dataloader_drop_last=DATALOADER_CONFIG["dataloader_drop_last"],
    
    
    num_train_epochs=TRAIN_CONFIG["num_train_epochs"],
    learning_rate=TRAIN_CONFIG["learning_rate"],
    weight_decay=TRAIN_CONFIG["weight_decay"],
    use_cpu=TRAIN_CONFIG["use_cpu"],
    
    evaluation_strategy=EVAL_CONFIG["eval_strategy"],
    eval_steps=EVAL_CONFIG["eval_steps"],
    
    save_strategy=CALLBACK_CONFIG["save_strategy"],
    output_dir=CALLBACK_CONFIG["output_dir"],
    
    run_name=wandb.run.name,
    report_to="wandb",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
