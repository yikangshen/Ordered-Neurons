import torch
import torch.nn as nn

import ON_LSTM


class ListOpsModel(nn.Module):
    def __init__(self, args):
        super(ListOpsModel, self).__init__()

        self.args = args
        self.padding_idx = args.padding_idx
        self.embedding = nn.Embedding(args.ntoken, args.ninp,
                                      padding_idx=self.padding_idx)

        self.encoder = ON_LSTM.ONLSTMStack(
            layer_sizes=[args.ninp] + [args.nhid] * args.nlayers,
            chunk_size=args.chunk_size,
            dropout=args.dropout,
            dropconnect=args.wdrop,
            std=args.std
        )

        self.mlp = nn.Sequential(
            nn.Dropout(args.dropouto),
            nn.Linear(args.nhid * 2 if args.bidirection else args.nhid, args.nout),
        )

        self.drop_input = nn.Dropout(args.dropouti)
        self.drop_output = nn.Dropout(args.dropouto)
        self.cost = nn.CrossEntropyLoss()

    def forward(self, input):
        input.transpose_(0,1)
        bsz = input.size(1)
        mask = (input != self.padding_idx).byte()
        lengths = mask.sum(0)

        emb = self.embedding(input)
        emb = self.drop_input(emb)
        hidden = self.encoder.init_hidden(bsz)
        raw_output, _, _, _, _ = self.encoder(emb, hidden)
        output = raw_output[lengths - 1, torch.arange(bsz).long()]
        output = self.mlp(output)
        return output

    def set_pretrained_embeddings(self, ext_embeddings, ext_word_to_index, word_to_index, finetune=False):
        assert hasattr(self, 'embedding')
        embeddings = self.embedding.weight.data.cpu().numpy()
        for word, index in word_to_index.items():
            if word in ext_word_to_index:
                embeddings[index] = ext_embeddings[ext_word_to_index[word]]
        embeddings = torch.from_numpy(embeddings).to(self.embedding.weight.device)
        self.embedding.weight.data.set_(embeddings)
        self.embedding.weight.requires_grad = finetune
