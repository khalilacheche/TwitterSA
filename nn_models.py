from torch.nn import Dropout, Linear, ReLU, Module, CrossEntropyLoss
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np


class PretrainedModelParameters:
    def __init__(self, name, src, output_size):
        self.name = name
        self.src = src
        self.output_size = output_size
        self.tokenizer = AutoTokenizer.from_pretrained(src)
        self.max_len = 256

    def get_text_encoding(self, text):
        return self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.maxlen,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )


bertweet_model_params = PretrainedModelParameters(
    name="bertweet",
    src="vinai/bertweet-large",
    output_size=4096,
)
x_distil_bert_l6h256_model_params = PretrainedModelParameters(
    name="xtreme distil l6 h256 cased",
    src="microsoft/xtremedistil-l6-h256-cased",
    output_size=256,
)


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, df, pretrained_model_params=x_distil_bert_l6h256_model_params):
        self.df = df
        self.maxlen = 256
        self.pretrained_model_params = pretrained_model_params

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tweet = self.df["text"].iloc[index].split()
        tweet = " ".join(tweet)
        sentiment = int(self.df["label"].iloc[index])

        encodings = self.pretrained_model_params.get_text_encoding(tweet)

        return {
            "input_ids": encodings.input_ids.flatten(),
            "attention_mask": encodings.attention_mask.flatten(),
            "labels": torch.tensor(sentiment, dtype=torch.long),
        }


class TransferLearningClassifier(Module):
    def __init__(self, pretrained_model_params=x_distil_bert_l6h256_model_params):
        super(TransferLearningClassifier, self).__init__()
        self.pretrained_model_params = pretrained_model_params
        self.base_model = AutoModel.from_pretrained(pretrained_model_params.src)
        self.linear1 = Linear(pretrained_model_params.output_size, 512)
        self.relu1 = ReLU()
        self.drop1 = Dropout(0.25)
        self.linear2 = Linear(512, 2)
        self.relu2 = ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask, return_dict=False)[0]
        x = self.relu1(self.linear1(outputs))
        x = self.drop1(x)
        x = self.relu2(self.linear2(x))
        return x[:, 0, :]

    def start_train_loop(
        self, train_loader, valid_loader, num_epochs, lr=1e-5, device="cpu"
    ):
        criterion = CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        for epoch in range(num_epochs):
            # TRAIN
            self.train()
            train_loop = tqdm(train_loader)
            for batch in train_loop:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                output = self(input_ids, attention_mask)
                loss = criterion(output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.parameters(), max_norm=1.0)
                optimizer.step()

                train_loop.set_description(f"Training Epoch: {epoch}")
                train_loop.set_postfix(loss=loss.item())

            # VALIDATION
            self.eval()
            valid_loop = tqdm(valid_loader)
            for batch in valid_loop:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                output = self(input_ids, attention_mask)
                loss = criterion(output, labels)

                valid_loop.set_description(f"Validation Epoch: {epoch}")
                valid_loop.set_postfix(loss=loss.item())

    def get_predictions(self, test_df, device="cpu"):
        self.eval()
        res = []
        for i in range(len(test_df)):
            text = test_df.iloc[i]
            encodings = self.pretrained_model_params.get_text_encoding(text)

        with torch.no_grad():
            self.to(device)
            preds = self(
                encodings["input_ids"].to(device),
                encodings["attention_mask"].to(device),
            )
            preds = np.argmax(preds)
            output = preds.item()
            res.append(output)
        return res
