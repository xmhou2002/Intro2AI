import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformer_model import TransformerModel


def get_data(file="data\\translate.csv", nums=None):
    all_datas = pd.read_csv(file)
    en_datas = list(all_datas["english"])
    ch_datas = list(all_datas["chinese"])

    if nums == None:
        return en_datas, ch_datas
    else:
        return en_datas[:nums], ch_datas[:nums]

#en_word_2_index是二维列表，共data个子列表，每个对应该data所含词在vocabulary中的index构成的列表（即wordID）
class MyDataset(Dataset):
    def __init__(self, en_data, ch_data, en_word_2_index, ch_word_2_index):
        self.en_data = en_data
        self.ch_data = ch_data
        self.en_word_2_index = en_word_2_index
        self.ch_word_2_index = ch_word_2_index
#重载对MyDataset的索引运算符，输入data序数，返回中英文data对应的词ID列表
    def __getitem__(self, index):
        en = self.en_data[index]
        ch = self.ch_data[index]

        en_index = [self.en_word_2_index[i] for i in en]
        ch_index = [self.ch_word_2_index[i] for i in ch]

        return en_index, ch_index
    
#基于整个dataset来对batch data进行处理
#输入的batch_datas实际上是这些data已经处理好的一堆wordID列表
    def batch_data_process(self, batch_datas):
        global device
        en_index, ch_index = [], []
        en_len, ch_len = [], []

        for en, ch in batch_datas:
            en_index.append(en)
            ch_index.append(ch)
            en_len.append(len(en))
            ch_len.append(len(ch))

        max_en_len = max(en_len)
        max_ch_len = max(ch_len)
        #下面的过程是对传入的wordID作padding以及修饰：
        # en word在最后添加<PAD>对应的ID，使各个wordID列表等长
        # ch word在起始位置添加<BOS>对应的ID，在结束位置添加<EOS>对应的ID，最后同样padding一下（但每个ch word应该都是一个字构成，好像区别没有en大）
        #等长后，转为tensor加速运算
        en_index = [i + [self.en_word_2_index["<PAD>"]] * (max_en_len - len(i)) for i in en_index]
        ch_index = [
            [self.ch_word_2_index["<BOS>"]] + i + [self.ch_word_2_index["<EOS>"]] + [self.ch_word_2_index["<PAD>"]] * (
                    max_ch_len - len(i)) for i in ch_index]
        # TODO 拓展 - question 1 : 这里每一个batch中的句子都一样长吗？为什么？
        en_index = torch.tensor(en_index, device=device)
        ch_index = torch.tensor(ch_index, device=device)

        return en_index, ch_index
    
#重载len()函数
    def __len__(self):
        assert len(self.en_data) == len(self.ch_data)
        return len(self.ch_data)

#继承torch.nn中的Module，塞入参数把其中的embedding class以及lstm class实例化，
#embedding基于word2vec模型计算得到corpus中各个单词的embedding vector（分布式表示）
    #en_corpus_len表示en data构成的corpus中的单词数
    #参数encoder_embedding_num表示word的embedding vector的维度
#多层lstm构成的encoder，需要传入处理的embedding vector的维度以及encoder内部hidden layer层数
    #encoder_hidden_num表示encoder lstm模型的hidden layer层数
class Encoder(nn.Module):
    def __init__(self, encoder_embedding_num, encoder_hidden_num, en_corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(en_corpus_len, encoder_embedding_num)
        self.lstm = nn.LSTM(encoder_embedding_num, encoder_hidden_num, batch_first=True)

#要注意区分：此处的forward是由好几层hidden lstm layers构成的encoder整体的forward，其输出为最后一层hidden layer的输出
    #输入的是由 一个个仅wordID处为1的one-hot vector（或仅传入wordID，再自行转为one-hot vector）组成的列表 构成的datas tensor
    def forward(self, en_index):
        # TODO 基础 - task 1: 这里实现encoder的过程：首先获得en_index的词嵌入表示，然后使用lstm模型获得最终输出。
        #作为函数调用embedding，返回值为把en_index datas中的one-hot vector替换为计算得到的所需维度的embedding vector构成的二维tensor
        en_embedding = self.embedding(en_index)
        #作为函数调用lstm进行前向传播，传出值取为last hidding layer的输出
        _, encoder_hidden = self.lstm(en_embedding)
        #
        return encoder_hidden

#和encoder的初始化完全一致，仅在forward的输入值和返回值处有差别，下面对此作了解释
class Decoder(nn.Module):
    def __init__(self, decoder_embedding_num, decoder_hidden_num, ch_corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(ch_corpus_len, decoder_embedding_num)
        self.lstm = nn.LSTM(decoder_embedding_num, decoder_hidden_num, batch_first=True)

#输入为已经得到的上一个decoder的words output（或在第一个encoder hidden layer中人为的words input）和上一个decoder输出的hidden；输出与之对应

    def forward(self, decoder_input, hidden):
        # TODO 基础 - task 2: 实现decoder的过程: 首先获得decoder_input的词嵌入表示，然后使用lstm模型获得最终输出。
        embedding = self.embedding(decoder_input)
        decoder_output, decoder_hidden = self.lstm(embedding, hidden)
        # 
        return decoder_output, decoder_hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder_embedding_num, encoder_hidden_num, en_corpus_len, decoder_embedding_num,
                 decoder_hidden_num, ch_corpus_len):
        super().__init__()
        self.encoder = Encoder(encoder_embedding_num, encoder_hidden_num, en_corpus_len)
        self.decoder = Decoder(decoder_embedding_num, decoder_hidden_num, ch_corpus_len)
        self.classifier = nn.Linear(decoder_hidden_num, ch_corpus_len)
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, en_index, ch_index):
        # TODO 拓展 - question 2: 为什么decoder_input舍弃最后一个字符，label舍弃第一个字符？
        decoder_input = ch_index[:, :-1]
        label = ch_index[:, 1:]

        # TODO 基础 - task 3: 实现seq2seq的过程。(调用之前实现的encoder,decoder)
        encoder_hidden = self.encoder(en_index)
        decoder_output, _ = self.decoder(decoder_input, encoder_hidden) 

        pre = self.classifier(decoder_output)
        loss = self.cross_loss(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))

        return loss


def translate(sentence):
    # TODO 拓展 - question 3: 训练过程和测试过程decoder的输入有何不同？
    global en_word_2_index, model, device, ch_word_2_index, ch_index_2_word
    en_index = torch.tensor([[en_word_2_index[i] for i in sentence]], device=device)

    result = []
    encoder_hidden = model.encoder(en_index)
    decoder_input = torch.tensor([[ch_word_2_index["<BOS>"]]], device=device)

    decoder_hidden = encoder_hidden
    while True:
        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
        pre = model.classifier(decoder_output)
#通过argmax函数取出输出的由 pre中最新生成的（dim=-1）一个个corpus_len维度的概率构成的vector 中最大的概率值处的index，参照中文corpus index的对应关系转为一个中文word
        w_index = int(torch.argmax(pre, dim=-1))
        word = ch_index_2_word[w_index]

        if word == "<EOS>" or len(result) > 50:
            break

        result.append(word)
        decoder_input = torch.tensor([[w_index]], device=device)

    print("译文: ", "".join(result))


'''
以上为基础实验内容部分
拓展部分基于attntion的翻译任务代码如下:
'''


class Attention_encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)

    def forward(self, en_index):
        embedded = self.embedding(en_index)
        encoder_output, encoder_hidden = self.rnn(embedded)
        encoder_hidden = encoder_hidden.squeeze(0)
        state = torch.tanh(encoder_hidden)
        return encoder_output, state


class Attention(nn.Module):
    def __init__(self, encoder_hidden_num, decoder_hidden_num):
        super().__init__()
        self.attn = nn.Linear(encoder_hidden_num + decoder_hidden_num, decoder_hidden_num)
        self.v = nn.Linear(decoder_hidden_num, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        en_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(en_len, 1, 1).transpose(0, 1)
        value = 0
        # TODO 拓展 task a: 实现attention的计算。
        energy = torch.tanh(self.attn(torch.cat((hidden,encoder_outputs), dim=2)))
        attention_v = self.v(energy).squeeze(2)
        value = F.softmax(attention_v, dim=-1)
        #
        return value


class Attention_decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU(encoder_hidden_dim + embedding_dim, decoder_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(encoder_hidden_dim + decoder_hidden_dim + embedding_dim, output_dim)

    def forward(self, decoder_input, s, encoder_output):
        decoder_input = decoder_input.unsqueeze(1)
        embedded = self.embedding(decoder_input).transpose(0, 1)
        # TODO 拓展 task b: 构造decoder的输入
        a = self.attention(s, encoder_output).unsueeze(1)
        c = torch.bmm(a, encoder_output).transpose(0,1)
        rnn_input = torch.cat((embedded,c), dim=2).transpose(0,1)
        #
        decoder_output, decoder_hidden = self.rnn(rnn_input, s.unsqueeze(0))
        embedded = embedded.squeeze(0)
        decoder_output = decoder_output.squeeze(1)
        c = c.squeeze(0)
        pred = self.fc_out(torch.cat((decoder_output, c, embedded), dim=1))
        return pred, decoder_hidden.squeeze(0)


class Seq2seq_Attention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.loss = nn.CrossEntropyLoss()

    def forward(self, en_index, ch_index):
        batch_size = en_index.shape[0]
        ch_len = ch_index.shape[1]
        ch_corpus_len = self.decoder.output_dim

        output = torch.zeros(batch_size, ch_len - 1, ch_corpus_len).to(self.device)
        encoder_ouput, encoder_hidden = self.encoder(en_index)
        decoder_input = ch_index[:, 0]
        label = ch_index[:, 1:]
        for i in range(0, ch_len - 1):
            decoder_output, state = self.decoder(decoder_input, encoder_hidden, encoder_ouput)
            output[:, i, :] = decoder_output
            decoder_input = ch_index[:, i + 1]
        loss = self.loss(output.reshape(-1, output.shape[-1]), label.reshape(-1))
        return loss

    def translate(self, sentence, en_word_2_index, ch_index_2_word):
        en_index = torch.tensor([[en_word_2_index[i] for i in sentence]], device=device)
        decoder_input = torch.tensor([ch_word_2_index["<BOS>"]], device=device)
        result = []
        encoder_ouput, encoder_hidden = self.encoder(en_index)
        while True:
            decoder_output, state = self.decoder(decoder_input, encoder_hidden, encoder_ouput)
            top1 = decoder_output.argmax(1)
            word = ch_index_2_word[top1]
            if word == "<EOS>" or len(result) > 50:
                break
            result.append(word)
            decoder_input = top1
        print("译文: ", "".join(result))


'''
拓展部分:
实现评测函数。
'''
def cal_blue(src_sentence, dst_sentence):
    # TODO 拓展 task c: 给定原文 src_sentence 和译文 dst_sentence 实现 BLUE 值计算，返回 BLUE 值
    pass

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Loading data...")
    with open("data\\ch.vec", "rb") as f1:
        _, ch_word_2_index, ch_index_2_word = pickle.load(f1)

    with open("data\\en.vec", "rb") as f2:
        _, en_word_2_index, en_index_2_word = pickle.load(f2)

    ch_corpus_len = len(ch_word_2_index)
    en_corpus_len = len(en_word_2_index)

    ch_word_2_index.update({"<PAD>": ch_corpus_len, "<BOS>": ch_corpus_len + 1, "<EOS>": ch_corpus_len + 2})
    en_word_2_index.update({"<PAD>": en_corpus_len})

    ch_index_2_word += ["<PAD>", "<BOS>", "<EOS>"]
    en_index_2_word += ["<PAD>"]

    ch_corpus_len += 3
    en_corpus_len = len(en_word_2_index)
    print("The length of ch_corpus_len is: ", ch_corpus_len)
    print("The length of en_corpus_len is: ", en_corpus_len)

    #
    # 设置参数:
    en_datas, ch_datas      = get_data(nums=1000)       # 注: 此处可以设置数据条数
                                                        #     可以在调试时选用较少的参数量
                                                        #     设为 None 或者去掉这个参数即为默认全量数据
    encoder_embedding_num   = 128
    encoder_hidden_num      = 128
    decoder_embedding_num   = 128
    decoder_hidden_num      = 128
    dropout         = 0.1
    nheads          = 8
    num_layers      = 2

    batch_size      = 32
    epochs          = 400
    print_freq      = 50
    lr              = 1e-3

    model_type_id   = 0
    # 0: vanilla Seq2Seq
    # 1: Attention Seq2Seq
    # 2: Transformer Seq2Seq
    model_type      = ["vanilla_seq2seq", 
                       "attention_seq2seq", 
                       "transformer_seq2seq"][model_type_id]

    print("Constructing dataloader...")
    dataset = MyDataset(en_datas, ch_datas, en_word_2_index, ch_word_2_index)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=dataset.batch_data_process)
    print("Dataloader constructed.")

    if model_type == "vanilla_seq2seq":  # Use vanilla Seq2Seq
        model = Seq2Seq(encoder_embedding_num, encoder_hidden_num, en_corpus_len, decoder_embedding_num,
                        decoder_hidden_num,
                        ch_corpus_len)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    elif model_type == "attention_seq2seq":  # Use Attention Seq2Seq
        attn = Attention(encoder_hidden_num, decoder_hidden_num)
        encoder = Attention_encoder(input_dim=en_corpus_len, embedding_dim=encoder_embedding_num,
                                    encoder_hidden_dim=encoder_hidden_num, decoder_hidden_dim=decoder_hidden_num)
        decoder = Attention_decoder(output_dim=ch_corpus_len, embedding_dim=decoder_embedding_num,
                                    encoder_hidden_dim=decoder_hidden_num, decoder_hidden_dim=decoder_hidden_num,
                                    attention=attn)
        model = Seq2seq_Attention(encoder, decoder, device)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    elif model_type == "transformer_seq2seq":  # Use Transformer Seq2Seq (第一次实验不需要考虑)
        model = TransformerModel(ch_corpus_len, encoder_embedding_num, nheads, encoder_hidden_num,
                                 num_layers, dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.NLLLoss()

    # train
    print("\nTraining...")
    total_loss = 0
    for epoch in range(epochs):
        print("\nEpoch: ", epoch + 1)
        for step, (en_index, ch_index) in enumerate(dataloader):
            en_index = en_index.to(device)
            ch_index = ch_index.to(device)
            if model_type == "transformer_seq2seq":
                output = model(en_index, ch_index[:, 1:])
                output = output.view(-1, ch_corpus_len)
                ch_index = ch_index[:, 1:].reshape(-1)
                loss = criterion(output, ch_index)
            else:
                loss = model(en_index, ch_index)
            total_loss += loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()
            if (step + 1) % print_freq == 0:
                print("  Step: ", step + 1, " Loss: ", loss.item())
        print("Epoch: ", epoch + 1, " Loss: ", total_loss / len(dataloader))
        total_loss = 0
    print("Training finished.")

    # save model
    print("Saving model...")
    torch.save(model.state_dict(), "ckpt\\ckpt.pkl")
    print("Model saved.")

    # generate:
    while True:
        s = input("请输入英文: ")
        if model_type == "vanilla_seq2seq":
            translate(s)
        else:
            model.translate(s, en_word_2_index, ch_index_2_word)
