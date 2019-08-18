import argparse
import time

import torch
import numpy as np

import preprocess
from preprocess import Vocab,Context,Dataset
import tempfile
from bandit import return_record_index,generate_reward
from beam_decode import beam_decode

np.set_printoptions(precision=4, suppress=True)


def greedy_sample(probs,
                   max_num_of_rec=3,device='cpu'):
    # sample sentences
    probs_numpy = probs.data.cpu().numpy()
    probs_numpy = np.reshape(probs_numpy, len(probs_numpy))
    greedy_index_list,_ = return_record_index(probs_numpy,probs,sample_method="greedy",max_num_of_rec=max_num_of_rec,device=device)
    return greedy_index_list




def ext_model_eval(model1,model2, vocab, args, eval_data="test",device='cpu'):
    print("loading data %s" % eval_data)

    model1.eval()
    # model2.eval()
    data_loader = preprocess.PickleReader(args.data_dir)
    eval_rewards = []
    data_iter = data_loader.chunked_data_reader(eval_data)
    print("doing model evaluation on %s" % eval_data)

    step_in_epoch=0
    for phase, dataset in enumerate(data_iter):
        for step, contexts in enumerate(preprocess.BatchDataLoader(dataset, shuffle=False)):
            print("Done %2d chunck, %4d/%4d context\r" % (phase+1, step + 1, len(dataset)),end=' ')
            
            context = contexts[0]
            records = context.records
            target = context.summary
            records = torch.autograd.Variable(torch.LongTensor(records)).to(device)
            target = torch.autograd.Variable(torch.LongTensor(target)).to(device)
            target_len = len(target)
            prob,r_cs = model1(records)
            cp = greedy_sample(prob,context.num_of_records,device)

            # ## greedy decode/ beam decode ?

            # gen_input = torch.autograd.Variable(r_cs[cp[0]]).to(device)
            # e_k,prev_hidden, prev_emb = model2(gen_input)
            # z_k = torch.autograd.Variable(records[cp[0]][0]).to(device)

            # gen_summary = beam_decode(model2,vocab,prev_hidden,gen_input,e_k,z_k)

            # summary = [vocab.index_to_word[i] for i in gen_summary]
            # gold_summary = [vocab.index_to_word[i] for i in context.summary]

            reward = generate_reward(None,None,gold_cp=context.gold_index,cp=cp)

            eval_rewards.append(reward)
            step_in_epoch+=1
    avg_eval_r = np.mean(eval_rewards, axis=0)
    print('model reward in %s:' % (eval_data))
    print(avg_eval_r)
    return avg_eval_r


if __name__ == '__main__':
    import pickle

    torch.manual_seed(1234)
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_file_1', type=str, default='rotowire/pickle_data/vocab.600d.src.p')
    parser.add_argument('--vocab_file_2', type=str, default='rotowire/pickle_data/vocab.600d.target.p')
    parser.add_argument('--data_dir', type=str, default='rotowire/pickle_data/')
    parser.add_argument('--data', type=str, default='test')
    parser.add_argument('--model_file_1', type=str, default='model1/bandit.model')
    parser.add_argument('--model_file_2', type=str, default='model2/gen.model')
    parser.add_argument('--device', type=int, default=0,
                        help='select GPU')
    parser.add_argument('--rl_baseline_method', type=str, default="greedy",
                        help='greedy, global_avg,batch_avg,or none')
    parser.add_argument('--length_limit', type=int, default=-1,
                        help='length limit output')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('generate config')
    with open(args.vocab_file_1, "rb") as f:
        vocab1 = pickle.load(f)
    with open(args.vocab_file_2, "rb") as f:
        vocab2 = pickle.load(f)

    print("loading existing models: %s " % args.model_file_1)
    model1 = torch.load(args.model_file_1, map_location=lambda storage, loc: storage)
    # model2 = torch.load(args.model_file_2, map_location=lambda storage, loc: storage)
    model1.cuda()
    # model2.cuda()
    data = args.data
    start_time = time.time()
    ext_model_eval(model1,None,vocab2,args,eval_data=data,device=device)
    print('Test time:', time.time() - start_time)
