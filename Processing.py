from transformers import Trainer, TrainingArguments, AutoModelForTokenClassification
from TransferData import train_dataset, dev_dataset, test_dataset, tokenizer, data_collator
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

# Assuming `all_labels` is a set of all unique labels in your dataset
# Make sure this is defined before you load the model
all_labels = set()
for dataset in [train_dataset, dev_dataset, test_dataset]:
    for entry in dataset:
        all_labels.update(entry['labels'])

unique_labels = list(all_labels)  # Make sure this is correctly populated
num_labels = len(unique_labels)

# id2label = {i: label for i, label in enumerate(label_list)}
# label2id = {label: i for i, label in enumerate(label_list)}

# Load the pre-trained model with the correct number of labels
model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-base-cased-finetuned-conll03-english",
    num_labels=num_labels,
    id2label={i: label for i, label in enumerate(unique_labels)},
    label2id={label: i for i, label in enumerate(unique_labels)},
    ignore_mismatched_sizes=True  # Not recommended
)


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Define the compute_metrics function for evaluation
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Convert the predictions and labels to the correct format for seqeval
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "classification_report": classification_report(true_labels, true_predictions),
    }

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

print(results)
