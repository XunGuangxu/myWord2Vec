import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import math
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

USE_CUDA = True
DATA_DIR = '/media/guangxu/O_o/UB/research/dataset/20newsgroups/'

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class Corpus:
    def __init__(self, corpus_pt):
        self.corpus_pt = corpus_pt
        self.w2id = {}
        self.id2w = {}
        self.w2cnt = {}
        self.docs = []
        self.process_corpus(corpus_pt)
    
    def process_corpus(self, corpus_pt):
        for line in open(corpus_pt):
            self.process_doc(line.strip().split())
            
    def process_doc(self, words):
        doc = []
        for word in words:
            if word not in self.w2id:
                self.w2id[word] = len(self.w2id)
                self.id2w[len(self.id2w)] = word
                self.w2cnt[word] = 1
            else:
                self.w2cnt[word] += 1
            doc.append(self.w2id[word])
        self.docs.append(doc)
        
    def write_vocab(self, vocab_pt):
        output_pt = open(vocab_pt, 'w')
        for i in range(len(self.id2w)):
            output_pt.writelines(self.id2w[i] + '\n')
        output_pt.writelines('<padding>\n')
        output_pt.close()
    
    def write_docInNum(self, output_doc_pt):
        output_pt = open(output_doc_pt, 'w')
        for doc in self.docs:
            output_pt.writelines(' '.join(map(str, doc)))
            output_pt.writelines('\n')
        output_pt.close
    

class myCBOWNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_neg):
        super(myCBOWNS, self).__init__()
        self.embeddings = nn.Embedding(vocab_size + 1, embedding_dim) # one more for padding
        self.embeddings.weight = nn.Parameter(torch.FloatTensor(vocab_size+1, embedding_dim).uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim))        
        self.vocab_size = vocab_size
        self.n_neg = n_neg
        self.sampling_weights = None
    
    def forward(self, target_wids, context_wids):
        batch_size = len(target_wids)        
        var_context_wids = Variable(context_wids)
        var_target_wids = Variable(target_wids)
        if USE_CUDA:
            var_context_wids = var_context_wids.cuda()
            var_target_wids = var_target_wids.cuda()        
            
        context_embeddings = self.embeddings(var_context_wids)
        avg_ctxt_embeddings = context_embeddings.mean(dim=1).unsqueeze(2)
        target_embeddings = self.embeddings(var_target_wids).unsqueeze(1)
        var_neg_wids = Variable(torch.FloatTensor(batch_size, self.n_neg).uniform_(0, self.vocab_size-1).long())
        if USE_CUDA:
            var_neg_wids = var_neg_wids.cuda()
        neg_embeddings = self.embeddings(var_neg_wids)
        
        pos_loss = torch.bmm(target_embeddings, avg_ctxt_embeddings).sigmoid().log().sum()
        neg_loss = torch.bmm(neg_embeddings.neg(), avg_ctxt_embeddings).sigmoid().log().sum()
        
        return -(pos_loss + neg_loss)
    
    
class CBOWData(Dataset):
    def __init__(self, data_pt):
        self.data = pickle.load(open(data_pt, 'rb'))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        target_wid, context_wids = self.data[idx]
        return target_wid, context_wids
    
    def getvocabsize(self):
        vocab_size = 0
        for target_wid, context_wids in self.data:
            vocab_size = max(vocab_size, target_wid)
        return vocab_size + 1
        

def preprocess_data(corpus_pt, data_pt, context_size):
    data_corpus = Corpus(corpus_pt)
    #data_corpus.write_vocab(DATA_DIR + 'result/myWord2Vec/vocab.txt')
    #data_corpus.write_docInNum(DATA_DIR + 'result/myWord2Vec/docInNum.txt')
    vocab_size = len(data_corpus.w2id)
    data = []
    for docid, doc in enumerate(data_corpus.docs):
        for target_idx, target_wid in enumerate(doc):
            context_idx_lo = max(0, target_idx-context_size)
            context_idx_hi = min(len(doc), target_idx+context_size+1)
            context_wids = doc[context_idx_lo: target_idx] + doc[target_idx+1: context_idx_hi]            
            context_wids.extend([vocab_size] * (2*context_size - len(context_wids))) #padding here
            data.append((target_wid, context_wids))
    pickle.dump(data, open(data_pt, 'wb'))
    print('preprocessing done')
            
        

corpus_pt = DATA_DIR + 'CoEmbedding/20news_min_cnt.txt'
data_pt = DATA_DIR + 'result/myWord2Vec/train.dat'
embedding_save_pt = DATA_DIR + 'result/myWord2Vec/myCBOW.npz'
embedding_dim = 50
n_epochs = 10
context_size = 5
n_neg = 10
batch_size = 256
FIRST_TIME = False

if FIRST_TIME:
    preprocess_data(corpus_pt, data_pt, context_size)

dataset = CBOWData(data_pt)
vocab_size = dataset.getvocabsize()
losses = []
model = myCBOWNS(vocab_size, embedding_dim, n_neg)
if USE_CUDA:
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()
for epoch in range(n_epochs):
    print('%dth epoch...' % (epoch))
    total_loss = 0.0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batchid, (target_wid, context_wids) in enumerate(dataloader):
        model.zero_grad()
        loss = model(target_wid, torch.stack(context_wids, dim=1))
        loss.backward()
        optimizer.step()       
        total_loss += loss.data
        
    losses.append(total_loss[0])
    print('time cost: ' + timeSince(start_time))
    np.savez(embedding_save_pt, C=model.embeddings.weight.data.cpu().numpy())
    

print(losses)
#print(data_corpus.id2w[535])
#print(data_corpus.w2id['sense'])
#print(data_corpus.id2w[1])
#print(data_corpus.w2id['atheism'])
#print(data_corpus.id2w[data_corpus.docs[10][1]])

    
'''

data_corpus = Corpus(corpus_pt)
b_target_wids, b_context_wids = [], [] #batch
start_time = time.time()
for epoch in range(n_epochs):
    print('%dth epoch...' % (epoch))
    total_loss = 0.0
    for docid, doc in enumerate(data_corpus.docs):
        for target_idx, target_wid in enumerate(doc):
            context_idx_lo = max(0, target_idx-context_size)
            context_idx_hi = min(len(doc), target_idx+context_size+1)
            context_wids = doc[context_idx_lo: target_idx] + doc[target_idx+1: context_idx_hi]
            
            context_wids.extend([vocab_size] * (10 - len(context_wids))) #padding here
            
            b_target_wids += [target_wid]
            b_context_wids += [context_wids]
            
            if len(b_target_wids) == batch_size:
                model.zero_grad()
                loss = model(b_target_wids, b_context_wids)
                loss.backward()
                optimizer.step()
                b_target_wids, b_context_wids = [], []            
                total_loss += loss.data
                break
        break
    break
            
    losses.append(total_loss[0])
'''
 