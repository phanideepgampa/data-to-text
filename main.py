import os
import argparse
import logging
import random
import pickle
from collections import namedtuple
import time
import traceback

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
# from lr_scheduler import ReduceLROnPlateau

import model
import evaluate
from preprocess import PickleReader,BatchDataLoader,Vocab,Context,Dataset
from bandit import ContextualBandit

np.set_printoptions(precision=4, suppress=True)

Config1 = namedtuple('parameters',
                    ['vocab_size', 'embedding_dim',
                     'LSTM_hidden_units','LSTM_layers','train_embed',
                     'word2id', 'id2word',
                     'dropout'])
Config2 = namedtuple('parameters',
                    ['vocab_size', 'embedding_dim',
                     'LSTM_hidden_units','LSTM_layers','train_embed','decode_type',
                     'word2id', 'id2word',
                     'dropout'])

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_model(args,vocab1,vocab2,device):
    print(args)
    print("generating config")
    config1 = Config1(
        vocab_size=len(vocab1),
        embedding_dim=args.embedding_dim,
        LSTM_layers= args.lstm_layer_1,
        LSTM_hidden_units= args.hidden,
        train_embed = args.train_embed,
        # pretrained_embedding=vocab1.embedding,
        word2id=vocab1.word_to_index,
        id2word=vocab1.index_to_word,
        dropout=args.dropout
    )
    config2 = Config2(
        vocab_size=len(vocab2),
        embedding_dim=args.embedding_dim,
        LSTM_layers= args.lstm_layer_2,
        LSTM_hidden_units= args.hidden,
        train_embed = args.train_embed,
        # pretrained_embedding=vocab2.embedding,
        word2id=vocab2.word_to_index,
        id2word=vocab2.index_to_word,
        dropout=args.dropout,
        decode_type = args.decode_type
    )
    model_name_1 = ".".join((args.model_file_1,
                           str(args.rl_baseline_method),args.sampling_method,
                           "gamma",str(args.gamma),
                           "beta",str(args.beta),
                           "batch",str(args.train_batch),
                           "learning_rate",str(args.lr_1),
                           "bsz", str(args.batch_size), 
                           "data", args.data_dir.split('/')[0],
                           "emb", str(config1.embedding_dim),
                           "dropout", str(args.dropout),
                           "max_num",str(args.max_num_of_ans),
                           "train_embed",str(args.train_embed),
                           'd2s'))
    # model_name_2 = ".".join((args.model_file_2,
    #                        "gamma",str(args.gamma),
    #                        "beta",str(args.beta),
    #                        "batch",str(args.train_batch),
    #                        "learning_rate",str(args.lr_2),
    #                        "data", args.data_dir.split('/')[0],
    #                        "emb", str(config2.embedding_dim),
    #                        "dropout", str(args.dropout),
    #                        'decode_type',str(args.decode_type),
    #                        'd2s'))


    log_name = ".".join(("log/model",
                           str(args.rl_baseline_method), args.sampling_method,
                           "gamma",str(args.gamma),
                           "beta",str(args.beta),
                            "batch",str(args.train_batch),
                           "lr_1",str(args.lr_1),"lr_2",str(args.lr_1),args.sampling_method,
                           "bsz", str(args.batch_size), 
                           "data", args.data_dir.split('/')[0],
                           "emb1", str(config1.embedding_dim),
                           "emb2", str(config2.embedding_dim),
                           "dropout", str(args.dropout),
                           'decode_type',str(args.decode_type),
                           "train_embed",str(args.train_embed),
                           'd2s'))

    print("initialising data loader and RL learner")
    data_loader = PickleReader(args.data_dir)
    data = args.data_dir.split('/')[0]
    num_data = 3398

    # init statistics
    reward_list = []
    loss_list1 =[]
    loss_list2 = []
    best_eval_reward = 0.
    model_save_name_1 = model_name_1
    # model_save_name_2 = model_name_2

    bandit = ContextualBandit(b=args.batch_size,rl_baseline_method=args.rl_baseline_method,vocab=vocab2,sample_method=args.sampling_method,device=device)

    print("Loaded the Bandit")
 
    model1 = model.Bandit(config1).to(device)
    # model2 = model.Generator(config2).to(device)
    print("Loaded the models")

    if args.load_ext:
        model_name_1 = args.model_file_1
        model_name_2 = args.model_file_2
        model_save_name_1 = model_name_1
        model_save_name_2 = model_name_2
        print("loading existing models:1->%s" % model_name_1)
        print("loading existing models:2->%s" % model_name_2)
        model1 = torch.load(model_name_1, map_location=lambda storage, loc: storage)
        model1.to(device)
        # model2 = torch.load(model_name_2, map_location=lambda storage, loc: storage)
        # model2.to(device)       
 
        print("finish loading and evaluate models:")
        # evaluate.ext_model_eval(extract_net, vocab, args, eval_data="test")
        best_eval_reward = evaluate.ext_model_eval(model1, model2,vocab2, args, "val")

    logging.basicConfig(filename='%s.log' % log_name,
                        level=logging.DEBUG, format='%(asctime)s %(levelname)-10s %(message)s')
    # Loss and Optimizer
    optimizer1 = torch.optim.Adam([param for param in model1.parameters() if param.requires_grad == True ], lr=args.lr_1, betas=(args.beta, 0.999),weight_decay=1e-6)
    # optimizer2 = torch.optim.Adam([param for param in model2.parameters() if param.requires_grad == True ], lr=args.lr_2, betas=(args.beta, 0.999),weight_decay=1e-6)

    # if args.lr_sch ==1:    
    #     scheduler = ReduceLROnPlateau(optimizer_ans, 'max',verbose=1,factor=0.9,patience=3,cooldown=3,min_lr=9e-5,epsilon=1e-6)
    #     if best_eval_reward:
    #         scheduler.step(best_eval_reward,0)
    #         print("init_scheduler")
    # elif args.lr_sch ==2:
    #     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_ans,args.lr, args.lr_2, step_size_up=3*int(num_data/args.train_batch), step_size_down=3*int(num_data/args.train_batch), mode='exp_range', gamma=0.98,cycle_momentum=False)   
    print("starting training")
    start_time = time.time()
    n_step = 100
    gamma = args.gamma
    n_val = int(num_data/(7*args.train_batch))
    decode_loss = torch.nn.CrossEntropyLoss()
    with torch.autograd.set_detect_anomaly(True):
        for epoch in tqdm(range(args.epochs_ext),desc="epoch:"):
            train_iter = data_loader.chunked_data_reader("train", data_quota=args.train_example_quota)  #-1
            step_in_epoch = 0
            for dataset in train_iter:
                for step, contexts in tqdm(enumerate(BatchDataLoader(dataset, batch_size=args.train_batch,shuffle=True))):
                    try:
                        model1.train()
                        # model2.train()
                        step_in_epoch += 1                        
                        reward=0.
                        for context in contexts:
                            records = context.records
                            target = context.summary
                            records = torch.autograd.Variable(torch.LongTensor(records)).to(device)
                            # target = torch.autograd.Variable(torch.LongTensor(target)).to(device)
                            # target_len = len(target)
                            prob,r_cs = model1(records)
                            sample_content, greedy_cp = bandit.sample(prob,context,context.num_of_records)
                            # # apply data_parallel after this step
                            # sample_content.append((greedy_cp,0))
                            # gen_summaries = []
                            # total_loss = 0.
                            # for cp in [(greedy_cp.data,0)]:
                            #     gen_input = torch.autograd.Variable(r_cs[cp[0]].data).to(device)
                            #     e_k,prev_hidden, prev_emb = model2(gen_input,vocab2)
                            #     z_k = torch.autograd.Variable(records[cp[0]][:,0].data).to(device)  
                            #     prev_t =0
                            #     loss=0.
                            #     gen_summary =[]           
                            #     ## perform bptt here
                            #     for y_t in range(target_len):
                            #         p_out, prev_hidden = model2.forward_step(prev_emb,prev_hidden,gen_input,e_k,z_k)
                            #         topv,topi = p_out.topk(1)
                            #         gen_summary.append(topi)
                            #         prev_emb = model2.get_embedding(topi)
                            #         loss += decode_loss(p_out,target[y_t].unsqueeze(0))

                            #         if (y_t-prev_t)==50:
                            #             prev_t = y_t
                            #             loss.backward(retain_graph=True)
                            #             loss.detach()
                            #     if prev_t < target_len:
                            #         loss.backward()
                            #         loss.detach()
                            #     gen_summaries.append((gen_summary,cp[1]))
                            #     loss/=float(target_len)
                            #     total_loss+=loss
                            # optimizer2.step()
                            # optimizer2.zero_grad()
                            # total_loss/=len(sample_content)
                            bandit_loss, reward = bandit.calculate_loss(sample_content,context.gold_index,greedy_cp)
                            bandit_loss.backward()
                            optimizer1.step()
                            optimizer1.zero_grad()

                        reward_list.append(reward)
                        loss_list1.append(bandit_loss.data.cpu().numpy()[0])
                        # loss_list2.append(total_loss)

                        # if args.lr_sch==2:
                        #     scheduler.step()    
                        total_loss = 0.
                        # logging.info('Epoch %d Step %d Reward %.4f Loss1 %.4f Loss2 %.4f' % (epoch, step_in_epoch, reward,bandit_loss,total_loss))
                        logging.info('Epoch %d Step %d Reward %.4f Loss1 %.4f' % (epoch, step_in_epoch, reward,bandit_loss))

                    except Exception as e:
                        print(e)
                        traceback.print_exc()

                    if (step_in_epoch) % n_step == 0 or step_in_epoch != 0:
                        # logging.info('Epoch ' + str(epoch) + ' Step ' + str(step_in_epoch) +
                        #     ' reward: ' + str(np.mean(reward_list))+' loss1: ' + str(np.mean(loss_list1))+' loss2: ' + str(np.mean(loss_list2)))
                        logging.info('Epoch ' + str(epoch) + ' Step ' + str(step_in_epoch) +
                            ' reward: ' + str(np.mean(reward_list))+' loss1: ' + str(np.mean(loss_list1)))
                        reward_list = []
                        loss_list1 = []
                        # loss_list2=[]

                    if (step_in_epoch) % n_val == 0 and step_in_epoch != 0:
                        print("doing evaluation")
                        model1.eval()
                        # model2.eval()
                        #eval_reward = evaluate.ext_model_eval(mcan_cb, vocab, args, "test")
                        eval_reward = evaluate.ext_model_eval(model1,None, vocab2, args, "val",device)
                        
                        if  eval_reward > best_eval_reward:
                            best_eval_reward = eval_reward
                            print("saving models %s : with eval_reward:" % model_save_name_1, eval_reward)
                            logging.debug("saving models"+str(model_save_name_1)+" "+"with eval_reward:"+ str(eval_reward))
                            torch.save(model1, model_save_name_1)
                            # torch.save(model2,model_save_name_2)
                        print('epoch ' + str(epoch) + ' reward in validation: '
                            + str(eval_reward))
                        logging.debug('epoch ' + str(epoch) + ' reward in validation: '
                            + str(eval_reward))
                        logging.debug('time elapsed:'+str(time.time()-start_time))
            # if args.lr_sch ==1:
            #     mcan_cb.eval()
            #     eval_reward = evaluate.ext_model_eval(mcan_cb, vocab, args, "val")
            #     #eval_reward = evaluate.ext_model_eval(mcan_cb, vocab, args, "test")
            #     scheduler.step(eval_reward[0],epoch)
    return model1

def main():
    seed_everything()
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_file_1', type=str, default='rotowire/pickle_data/vocab.600d.src.p')
    parser.add_argument('--vocab_file_2', type=str, default='rotowire/pickle_data/vocab.600d.target.p')
    parser.add_argument('--data_dir', type=str, default='rotowire/pickle_data/')
    parser.add_argument('--model_file_1', type=str, default='model1/bandit.model')
    parser.add_argument('--model_file_2', type=str, default='model2/gen.model')
    parser.add_argument('--decode_type',type = str, default='conditional')
    parser.add_argument('--beta',type=float,default = 0.9)
    parser.add_argument('--gamma',type=float,default=0.9)
    parser.add_argument('--lr_sch',type=int,default=1)
    parser.add_argument('--lstm_layer_1',type=int,default=1)
    parser.add_argument('--lstm_layer_2',type=int,default=1)
    parser.add_argument('--train_embed',type=bool,default=False)
    parser.add_argument('--max_num_of_ans',type=int,default=5)
    parser.add_argument('--train_batch',type=int,default=1)
    parser.add_argument('--sampling_method',type=str,default = "herke")
    parser.add_argument('--epochs_ext', type=int, default=10)
    parser.add_argument('--load_ext', action='store_true')
    parser.add_argument('--hidden', type=int, default=600)
    parser.add_argument('--embedding_dim', type=int, default=600)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr_1', type=float, default=5e-5)
    parser.add_argument('--lr_2', type=float, default=8e-5)
    parser.add_argument('--device', type=int, default=0,
                        help='select GPU')
    parser.add_argument('--oracle_length', type=int, default=3,
                        help='-1 for giving actual oracle number of sentences'
                             'otherwise choose a fixed number of sentences')
    parser.add_argument('--rl_baseline_method', type=str, default="batch_avg",
                        help='greedy, global_avg, batch_avg, batch_med, or none')
    parser.add_argument('--rl_loss_method', type=int, default=2,
                        help='1 for computing 1-log on positive advantages,'
                             '0 for not computing 1-log on all advantages')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--train_example_quota', type=int, default=-1,
                        help='how many train example to train on: -1 means full train data')
    parser.add_argument('--prt_inf', action='store_true')

    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('generate config')
    with open(args.vocab_file_1, "rb") as f:
        vocab1 = pickle.load(f)
    with open(args.vocab_file_2, "rb") as f:
        vocab2 = pickle.load(f)

    model1,model2 = train_model(args, vocab1,vocab2,device)


if __name__ == '__main__':
    main()
    
