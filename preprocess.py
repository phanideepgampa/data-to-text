import numpy as np 
import random
import argparse
import pickle
import os
import linecache
from tqdm import tqdm 

"""
Modified/Adapted from https://github.com/yuedongP/BanditSum

"""
class Context():
    def __init__(self,records,summary,num = None,gold_index=None):
        self.records=records
        self.summary=summary
        self.num_of_records = num
        self.gold_index = gold_index

class Dataset():
    def __init__(self,context_list):
        self._data=context_list
    def __len__(self):
        return len(self._data)
    def __call__(self,batch_size,shuffle=True):
        total_size=len(self)
        if shuffle:
            random.shuffle(self._data)
        batches = [self._data[i:i+batch_size] for i in range(0,total_size,batch_size)]
        return batches
    def __getitem__(self,index):
        return self._data[index]

class Vocab():
    def __init__(self):
        self.word_list=['<pad>','<unk>','<sos>','<eos>']
        self.word_to_index={}
        self.index_to_word={}
        self.count=0
        self.embedding=None
    def __getitem__(self,key):
        if key in self.word_to_index.keys():
            return self.word_to_index[key]
        else:
            return self.word_to_index['<unk>']
    def __len__(self):
        return len(self.word_list)
    def add_vocab(self, vocab_file="data/vocab.txt"):
        with open(vocab_file, "r",encoding='utf-8') as f:
            for line in f:
                self.word_list.append(line.split()[0])  
        print("read %d words from vocab file" % len(self.word_list))
        for w in self.word_list:
            self.word_to_index[w] = self.count
            self.index_to_word[self.count] = w
            self.count += 1
    def add_embedding(self, mode=-1, embed_size=300):
        if mode ==0:
            return
        elif mode == -1:
            self.embedding = np.random.randn(len(self.word_list), embed_size)
        
        # print("Loading Glove embeddings")
        # with open(gloveFile, 'rb') as f:
        #     model = {}
        #     w_set = set(self.word_list)
        #     if init:
        #         embedding_matrix = np.random.randn(len(self.word_list), embed_size)
        #     else:
        #         embedding_matrix = np.zeros(shape=(len(self.word_list), embed_size))

        #     for line in f:
        #         splitLine = line.split()
        #         word = splitLine[0]
        #         if word in w_set:  # only extract embeddings in the word_list
        #             embedding = np.array([float(val) for val in splitLine[1:]])
        #             model[word] = embedding
        #             embedding_matrix[self.word_to_index[word]] = embedding
        #             if len(model) % 1000 == 0:
        #                 print("processed %d data" % len(model))
        # embedding_matrix[1] = np.random.multivariate_normal(np.zeros(embed_size),np.eye(embed_size)) # for unkown words
        # self.embedding = embedding_matrix
        # print("%d words out of %d has embeddings in the glove file" % (len(model), len(self.word_list)))

class BatchDataLoader():
    def __init__(self, dataset, batch_size=1, shuffle=True):
        assert isinstance(dataset, Dataset)
        assert len(dataset) >= batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return iter(self.dataset(self.batch_size, self.shuffle))
class PickleReader():
    """

    this class intends to read pickle files converted by RawReader

    """

    def __init__(self, pickle_data_dir="trec_qa/pickle_data/"):
        """
        :param pickle_data_dir: the base_dir where the pickle data are stored in
        this dir should contain train.p, val.p, test.p, and vocab.p
        this dir should also contain the chunked_data folder
        """
        self.base_dir = pickle_data_dir

    def data_reader(self, dataset_path):
        """
        :param dataset_path: path for data.p
        :return: data: Dataset objects (contain Document objects with doc.content and doc.summary)
        """
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
        return data

    def full_data_reader(self, dataset_type="train"):
        """
        this method read the full dataset
        :param dataset_type: "train", "val", or "test"
        :return: data: Dataset objects (contain Document objects with doc.content and doc.summary)
        """
        return self.data_reader(self.base_dir + dataset_type + ".p")

    def chunked_data_reader(self, dataset_type="train", data_quota=-1):
        """
        this method reads the chunked data in the chunked_data folder
        :return: a iterator of chunks of datasets
        """
        data_counter = 0
        # chunked_dir = self.base_dir + "chunked/"
        chunked_dir = os.path.join(self.base_dir, 'chunked')
        os_list = os.listdir(chunked_dir)
        if data_quota == -1: #none-quota randomize data
            random.seed()
            random.shuffle(os_list)

        for filename in os_list:
            if filename.startswith(dataset_type):
                # print("filename:", filename)
                chunk_data = self.data_reader(os.path.join(chunked_dir, filename))
                if data_quota != -1:  # cut off applied
                    quota_left = data_quota - data_counter
                    # print("quota_left", quota_left)
                    if quota_left <= 0:  # no more quota
                        break
                    elif quota_left > 0 and quota_left < len(chunk_data):  # return partial data
                        yield Dataset(chunk_data[:quota_left])
                        break
                    else:
                        data_counter += len(chunk_data)
                        yield chunk_data
                else:
                    yield chunk_data
            else:
                continue

def generate_vocab(vocab_file,outfile="rotowire/pickle_data/",mode = -1 ,embed_size=600,data="src"):
    vocab=Vocab()
    vocab.add_vocab(vocab_file)
    vocab.add_embedding(mode=mode,embed_size=embed_size)
    pickle.dump(vocab,open(outfile+'vocab.'+str(embed_size)+'d.'+data+'.p',"wb+"))

def write_to_pickle(in_file, out_file,vocab1,vocab2,chunk_size=1000,data='train'):
    b=1
    length =0 
    words = 0
    min_length = 10000000000
    max_length = -1
    num_records = []
    record_ids=[]
    if data in ['train','valid']:
        content_plan = open(os.path.join(in_file,data+'_content_plan.txt'),encoding='utf-8')
        for line in content_plan:
            temp = line.split()
            num_records.append(len(temp))
            record_ids.append([int(x)-4 for x in temp])
    else:
        content_plan = open(os.path.join(in_file,data+'_content_plan.txt'),encoding='utf-8')
        for line in content_plan:
            temp = line.split()
            num_records.append(len(temp)+4)
    contexts=[]
    records_list=[]
    with open(os.path.join(in_file,"src_"+data+'.txt'),encoding='utf-8') as records, open(os.path.join(in_file,"tgt_"+data+'.txt'),encoding='utf-8') as summaries:
        for r,s in tqdm(zip(records,summaries),desc=in_file) :
            temp = r.split()[4:]
            num_r = num_records[b-1]
            records_list=[]
            if record_ids:
                r_id = record_ids[b-1]
            else:
                r_id = None
            for feat in temp:
                records_list.append([vocab1[x] for x in feat.split('￨')])
            summary= []
            summary.extend([vocab2[y] for y in s.split()])
            summary.append(vocab2['<eos>'])
            length+=len(records_list)
            words+= len(summary)
            min_length = min(min_length,len(records_list))
            max_length = max(max_length,len(records_list))
            contexts.append(Context(records_list,summary,num_r,r_id))
            
            if b % chunk_size ==0 :
                pickle.dump(Dataset(contexts),open(out_file%(b/chunk_size),"wb+"))
                contexts=[]
            b+=1
    if contexts != []:
        pickle.dump(Dataset(contexts),open(out_file%(b/chunk_size+1),"wb+"))
        # pass
    print("No of contexts in %s file : %d"%(data,b))
    print("Avg number of  input records: %0.4f"%(length/float(b)))
    print("Avg number of  summary words: %0.4f"%(words/float(b)))
    print("Min number of  input records: %d"%(min_length))
    print("Max number of  input records: %d"%(max_length))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--vocabulary',type=bool,default=False)
    parser.add_argument('-v1','--vocabulary_src',type=str,default="rotowire/vocab_input.txt")
    parser.add_argument('-v2','--vocabulary_target',type=str,default="rotowire/vocab_output.txt")
    parser.add_argument('-emb_size','--embedding_size',type=int,default=600)
    parser.add_argument('-vo','--vocabulary_output',type=str,default="rotowire/pickle_data/")    
    # parser.add_argument('-emb','--embedding_file',type=str,default="glove/glove.6B.300d.txt")
    parser.add_argument('-p','--parse',type=bool,default=True)
    parser.add_argument('-p_train','--parse_train',type=str,default="rotowire/")
    parser.add_argument('-p_val','--parse_val',type=str,default="rotowire/")
    parser.add_argument('-p_test','--parse_test',type=str,default="rotowire/test/")
    parser.add_argument('-p_out','--parse_output',type=str,default="rotowire/pickle_data/chunked")

    args=parser.parse_args()
    os.chdir("C:/Users/Phanideep/Desktop/data-to-text")
    if args.vocabulary:
        vocab_src= args.vocabulary_src
        vocab_target = args.vocabulary_target
        # gloveFile=args.embedding_file
        generate_vocab(vocab_src,args.vocabulary_output,mode=0,embed_size=args.embedding_size,data="src")
        generate_vocab(vocab_target,args.vocabulary_output,mode=0,embed_size=args.embedding_size,data="target")
    if args.parse:
        train = args.parse_train
        val = args.parse_val
        test = args.parse_test
        with open("rotowire/pickle_data/vocab.600d.src.p",'rb') as f1, open("rotowire/pickle_data/vocab.600d.target.p",'rb') as f2:
            vocab1= pickle.load(f1)
            vocab2 = pickle.load(f2)
        write_to_pickle(test, os.path.join(args.parse_output,"test_%03d.bin.p"),vocab1,vocab2,chunk_size=200,data='test')
        write_to_pickle(val, os.path.join(args.parse_output,"val_%03d.bin.p"),vocab1,vocab2, chunk_size=200,data='valid')
        write_to_pickle(train, os.path.join(args.parse_output,"train_%03d.bin.p"),vocab1,vocab2,data='train',chunk_size=500)

if __name__=='__main__':
    main()
