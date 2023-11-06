from transformers import AutoTokenizer, DataCollatorForTokenClassification
from torch.utils.data import Dataset
import torch
#读取IOB文件的函数
def read_iob_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    words = []
    labels = []
    for line in lines:
        line = line.strip()  # 去除行尾的空白字符
        if not line:  # 如果是空行，表示句子的结束
            if words and labels:
                sentences.append({"words": words, "labels": labels})
                words = []
                labels = []
            continue

        # 检查行中是否有分隔符
        if ' ' in line:
            word, label = line.split(' ')
        elif '\t' in line:
            word, label = line.split('\t')
        else:
            continue  # 如果行中没有分隔符，跳过该行

        words.append(word)
        labels.append(label)

    # 添加最后一个句子，如果文件不是以空行结束的
    if words and labels:
        sentences.append({"words": words, "labels": labels})

    return sentences

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-cased-finetuned-conll03-english")
# 读取数据
data_train = read_iob_file('wnut17train.conll')
data_dev = read_iob_file('emerging.dev.conll')
data_test = read_iob_file('emerging.test.annotated')

# 获取所有的标签
all_labels = set()
for dataset in [data_train, data_dev, data_test]:
    for entry in dataset:
        all_labels.update(entry["labels"])

# 创建标签到ID的映射
label_to_id = {label: i for i, label in enumerate(all_labels)}

# 3. 定义自定义的Dataset类
class CustomDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels, max_length=128):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]["words"]
        label = self.labels[idx]["labels"]
        tokens = self.tokenizer(sentence, truncation=True, padding='max_length', is_split_into_words=True, return_tensors="pt")
        token_ids = tokens.input_ids.squeeze()
        attention_mask = tokens.attention_mask.squeeze()

        # Convert label names to ids
        label_ids = [-100] * self.max_length
        for i, (word, label_name) in enumerate(zip(sentence, label)):
            tokenized_word = tokenizer.tokenize(word)
            label_id = label_to_id[label_name]
            label_ids[i:i + len(tokenized_word)] = [label_id] * len(tokenized_word)

        return {
            "input_ids": token_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "labels": label_ids
        }

train_dataset = CustomDataset(tokenizer, data_train, data_train)
dev_dataset = CustomDataset(tokenizer, data_dev, data_dev)
test_dataset = CustomDataset(tokenizer, data_test, data_test)

data_collator = DataCollatorForTokenClassification(tokenizer)

