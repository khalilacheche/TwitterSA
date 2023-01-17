from torch.nn import Dropout, Linear, ReLU, Module, CrossEntropyLoss, LSTM
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

BASE_PATH = "drive/MyDrive/Twitter_SA-v1.2/"


class PretrainedModelParameters:
    def __init__(self, name, src, output_size):
        self.name = name
        self.src = src
        self.output_size = output_size
        self.tokenizer = AutoTokenizer.from_pretrained(src)
        self.max_len = 128

    def get_text_encoding(self, text):
        return self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            # Return PyTorch Tensors
            return_tensors="pt",
        )


bertweet_model_params = PretrainedModelParameters(
    name="bertweet_large",
    src="vinai/bertweet-large",
    output_size=1024,
)
x_distil_bert_l6h256_model_params = PretrainedModelParameters(
    name="xtreme_distilbert_l6_h256_uncased",
    src="microsoft/xtremedistil-l6-h256-uncased",
    output_size=256,
)


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        pretrained_model_params=x_distil_bert_l6h256_model_params,
        hasLabels=True,
    ):
        self.df = df
        self.pretrained_model_params = pretrained_model_params
        self.hasLabels = hasLabels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df["text"].iloc[index]
        length = len(self.df["text"].iloc[index].split(" "))
        if self.hasLabels:
            label = int(self.df["label"].iloc[index])
        encodings = self.pretrained_model_params.get_text_encoding(text)
        input_ids = encodings.input_ids.flatten()
        length = input_ids.size()[0] - input_ids.tolist().count(0)
        return {
            "length": torch.tensor(length, dtype=torch.int),
            "text": text,
            "input_ids": input_ids,
            "attention_mask": encodings.attention_mask.flatten(),
            "label": torch.tensor(label, dtype=torch.long) if self.hasLabels else 0,
        }


class TransferLearningClassifier(Module):
    def __init__(
        self,
        pretrained_model_params=x_distil_bert_l6h256_model_params,
        freeze_pretrained=True,
    ):
        super(TransferLearningClassifier, self).__init__()
        self.pretrained_model_params = pretrained_model_params
        self.base_model = AutoModel.from_pretrained(pretrained_model_params.src)
        # FT: dropout, architecture
        self.linear1 = Linear(2 * pretrained_model_params.output_size, 512)
        self.relu1 = ReLU()
        self.drop1 = Dropout(0.25)
        self.linear2 = Linear(512, 2)
        self.relu2 = ReLU()
        if freeze_pretrained:
            for name, param in self.named_parameters():
                if param.requires_grad and "base_model" in name:
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # first element is the sequence of word embeddings of returned tuple is of shape (batch_size, max_length(max number of tokens for document), embedding_dimension)
        # second element is the document embedding of shape (batch_size, embedding_dimension)
        word_embeddings, doc_embedding = self.base_model(
            input_ids, attention_mask, return_dict=False
        )
        # calculate the mean words embedding of the sequence of tokens in the model
        pooled_word_embeddings = torch.mean(word_embeddings, dim=1)

        # concatenate the document embedding and the mean words embedding
        x = torch.cat((doc_embedding, pooled_word_embeddings), dim=1)

        # feed through the last layers of the model
        x = self.relu1(self.linear1(x))
        x = self.drop1(x)
        x = self.relu2(self.linear2(x))
        return x

    def start_train_loop(
        self, train_loader, valid_loader, num_epochs, device, save_path, lr=1e-5
    ):
        training_stats = []
        validation_stats = []
        criterion = CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        for epoch in range(num_epochs):

            # Training phase

            self.train()  # set model to train mode
            epoch_train_loss = 0
            i = 1
            train_loader_iterator = tqdm(train_loader)

            for batch in train_loader_iterator:
                # reset gradient
                optimizer.zero_grad()

                # Get input
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                # Feed forward
                output = self(input_ids, attention_mask)

                # Calculate loss and update the model
                loss = criterion(output, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=1.0
                )  # Clip gradient to avoid exploding gradient
                optimizer.step()

                # Save and print out values
                epoch_train_loss += loss.item()
                current_mean = epoch_train_loss / i
                train_loader_iterator.set_description(f"Training Epoch: {epoch}")
                train_loader_iterator.set_postfix(loss=current_mean)
                i += 1

            epoch_train_loss = epoch_train_loss / len(train_loader)
            print("mean train loss for epoch", epoch, ":", epoch_train_loss)
            training_stats.append(epoch_train_loss)

            # Validation phase

            self.eval()  # set model to evaluation mode
            valid_loader_iterator = tqdm(valid_loader)
            epoch_validation_loss = 0
            i = 1
            for batch in valid_loader_iterator:
                # Get input
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                # Feed forward
                output = self(input_ids, attention_mask)

                # calculate loss
                loss = criterion(output, labels)

                # Save and print out values
                epoch_validation_loss += loss.item()
                current_mean = epoch_validation_loss / i
                valid_loader_iterator.set_description(f"Validation Epoch: {epoch}")
                valid_loader_iterator.set_postfix(loss=current_mean)
                i += 1
            epoch_validation_loss = epoch_validation_loss / len(valid_loader)
            print("mean valdiation loss for epoch", epoch, ":", epoch_validation_loss)
            validation_stats.append(epoch_validation_loss)

            # Save the model in between each epoch
            torch.save(
                self.state_dict(),
                save_path
                + (
                    "model_{}-epoch_{}".format(self.pretrained_model_params.name, epoch)
                ),
            )
        return training_stats, validation_stats

    def get_predictions(self, test_loader, device):
        self.eval()
        res = []
        test_loader_iterator = tqdm(test_loader)
        for batch in test_loader_iterator:
            with torch.no_grad():
                # move model to specified device (CPU or GPU)
                self.to(device)

                # Get input
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # feed forward
                output = self(
                    input_ids,
                    attention_mask,
                )

                # format output
                preds = output.argmax(dim=1)
                res = res + [
                    (batch["text"][i], preds[i].item())
                    for i in range(len(batch["text"]))
                ]

        return res


class BiLSTMTransferLearningClassifier(Module):
    def __init__(
        self,
        pretrained_model_params=x_distil_bert_l6h256_model_params,
        freeze_pretrained=True,
        lstm_units=64,
    ):
        super(BiLSTMTransferLearningClassifier, self).__init__()
        self.pretrained_model_params = pretrained_model_params
        self.freeze_pretrained = freeze_pretrained
        self.base_model = AutoModel.from_pretrained(pretrained_model_params.src)

        self.lstm = self.lstm = LSTM(
            (pretrained_model_params.output_size),
            lstm_units,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.linear1 = Linear(2 * lstm_units + pretrained_model_params.output_size, 256)
        self.relu1 = ReLU()
        self.linear2 = Linear(256, 2)
        self.relu2 = ReLU()
        if freeze_pretrained:
            for name, param in self.named_parameters():
                if param.requires_grad and "base_model" in name:
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, lengths):
        # first element is the sequence of word embeddings of returned tuple is of shape (batch_size, max_length(max number of tokens for document), embedding_dimension)
        # second element is the document embedding of shape (batch_size, embedding_dimension)
        word_embeddings, doc_embedding = self.base_model(
            input_ids, attention_mask, return_dict=False
        )
        # prepare the word embeddings for the LSTM
        packed_input = pack_padded_sequence(
            word_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # pass the sequence of word embeddings through the LSTM
        packed_output, (ht, ct) = self.lstm(packed_input)
        # reformat output
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        last_hidden_states = output[
            :, -1, :
        ]  # last hidden state of the LSTM unit after passsing all the sequences

        # Concatenate the document embedding and the last hidden state
        x = torch.cat((doc_embedding, last_hidden_states), dim=1)

        # Feed to the last layers of the model
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        return x

    def start_train_loop(
        self,
        train_loader,
        valid_loader,
        num_epochs,
        device,
        save_path,
        lr=1e-5,
    ):
        if self.freeze_pretrained and math.isclose(1e-5, lr, rel_tol=1e-6):
            lr = 5e-4
        training_stats = []
        validation_stats = []
        criterion = CrossEntropyLoss().to(device)
        optimizer = (
            torch.optim.AdamW(self.parameters(), lr=lr)
            if not self.freeze_pretrained
            else torch.optim.Adam(self.parameters(), lr=lr)
        )

        for epoch in range(num_epochs):

            # Training phase

            self.train()  # set model to train mode
            epoch_train_loss = 0
            train_loader_iterator = tqdm(train_loader)
            i = 1

            for batch in train_loader_iterator:
                # reset gradient
                optimizer.zero_grad()

                # Get input
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                lengths = batch["length"].to(device)

                # Feed forward
                output = self(input_ids, attention_mask, lengths)

                # Calculate loss and update the model
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                # Save and print out values
                epoch_train_loss += loss.item()
                current_mean = epoch_train_loss / i
                train_loader_iterator.set_description(f"Training Epoch: {epoch}")
                train_loader_iterator.set_postfix(loss=current_mean)
                i += 1
            epoch_train_loss = epoch_train_loss / len(train_loader)
            print("mean train loss for epoch", epoch, ":", epoch_train_loss)
            training_stats.append(epoch_train_loss)

            # Validation phase

            self.eval()  # set model to evaluation mode
            valid_loader_iterator = tqdm(valid_loader)
            epoch_validation_loss = 0
            i = 1

            for batch in valid_loader_iterator:
                # Get input
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                lengths = batch["length"].to(device)

                # Feed forward
                output = self(input_ids, attention_mask, lengths)

                # calculate loss
                loss = criterion(output, labels)

                # Save and print out values
                epoch_validation_loss += loss.item()
                current_mean = epoch_validation_loss / i
                valid_loader_iterator.set_description(f"Validation Epoch: {epoch}")
                valid_loader_iterator.set_postfix(loss=current_mean)
                i += 1
            epoch_validation_loss = epoch_validation_loss / len(valid_loader)
            print("mean valdiation loss for epoch", epoch, ":", epoch_validation_loss)
            validation_stats.append(epoch_validation_loss)

            # Save the model in between each epoch
            torch.save(
                self.state_dict(),
                save_path
                + "model_bilstm_{}_{}-epoch_{}".format(
                    "frozen" if self.freeze_pretrained else "unfrozen",
                    self.pretrained_model_params.name,
                    epoch,
                ),
            )
        return training_stats, validation_stats

    def get_predictions(self, test_loader, device):
        self.eval()
        res = []
        test_loader_iterator = tqdm(test_loader)
        for batch in test_loader_iterator:
            with torch.no_grad():
                # move model to specified device (CPU or GPU)
                self.to(device)

                # Get input
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                lengths = batch["length"].to(device)

                # feed forward
                output = self(input_ids, attention_mask, lengths)

                # format output
                preds = output.argmax(dim=1)
                res = res + [
                    (batch["text"][i], preds[i].item())
                    for i in range(len(batch["text"]))
                ]

        return res
