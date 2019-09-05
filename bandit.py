import random

import numpy as np
import torch
from torch.autograd import Variable
from copy import deepcopy

from nltk.translate.bleu_score import corpus_bleu
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from rank_metrics import mean_reciprocal_rank,average_precision,ndcg_at_k,dcg_at_k

# code modified/adapted from https://github.com/yuedongP/BanditSum




def return_record_index(probs_numpy, probs_torch, sample_method="sample", max_num_of_rec=10,method ="herke",device='cpu'):
    """
    :param probs: numpy array of the probablities for all records for a table
    :param sample_method: greedy or sample
    :param max_num_of_rec: max num of records to be selected
    :return: a list of index for the selected table
    """
    assert isinstance(sample_method, str)
    if max_num_of_rec <= 0:
        if sample_method == "sample":
            l = np.random.binomial(1, probs_numpy)
        elif sample_method == "greedy":
            l = [1 if prob >= 0.5 else 0 for prob in probs_numpy]
        record_index = np.nonzero(l)[0]
    else:
        if sample_method == "sample":
            probs_torch = probs_torch.squeeze()
            if len(probs_torch.size()) != 1:
                print(probs_torch)
                print(probs_torch.size()==torch.Size([]))

            if method == 'original':
                # original method
                probs_clip = probs_numpy * 0.8 + 0.1
                # print("sampling the index for the record")
                index = range(len(probs_clip))
                probs_clip_norm = probs_clip / sum(probs_clip)
                record_index = np.random.choice(index, max_num_of_rec, replace=False,
                                                 p=np.reshape(probs_clip_norm, len(probs_clip_norm)))
                p_record_index = probs_numpy[record_index]
                sorted_idx = np.argsort(p_record_index)[::-1]
                record_index = record_index[sorted_idx]
                loss = 0.
                for idx in index:
                    if idx in record_index:
                        loss += probs_torch[idx].log()
                    else:
                        loss += (1 - probs_torch[idx]).log()
            elif method == 'herke':
                # herke's method
                record_index = []
                epsilon = 0.1
                mask = Variable(torch.ones(probs_torch.size()).to(device), requires_grad=False)
                # mask = Variable(torch.ones(probs_torch.size()), requires_grad=False)
                loss_list = []
                for i in range(max_num_of_rec):
                    p_masked = probs_torch * mask
                    if random.uniform(0, 1) <= epsilon:  # explore
                        selected_idx = torch.multinomial(mask, 1)
                    else:
                        selected_idx = torch.multinomial(p_masked, 1)
                    loss_i = (epsilon / mask.sum() + (1 - epsilon) * p_masked[selected_idx] / p_masked.sum()).log()
                    loss_list.append(loss_i)
                    mask = mask.clone()
                    mask[selected_idx] = 0
                    record_index.append(selected_idx)

                record_index = torch.cat(record_index, dim=0)
                record_index = record_index.data.cpu().numpy()

                loss = sum(loss_list)
        elif sample_method == "greedy":
            loss = 0
            record_index = np.argsort(np.reshape(probs_numpy, len(probs_numpy)))[-max_num_of_rec:]
            record_index = record_index[::-1]

    # record_index.sort()
    return record_index, loss



def generate_reward(gold_summary, summary,gold_cp,cp,reward_type=1):
    #Bleu score
    # bleu = corpus_bleu([gold_summary],summary)

    cp = list(deepcopy(cp))
    # DLD
    if gold_cp:
        dld = normalized_damerau_levenshtein_distance(list(gold_cp),list(cp))
    else:
        dld=0.
    boolean = np.zeros(len(cp))
    for pos,element in enumerate(cp):
        if element in gold_cp:
            boolean[pos]=1
    precision = np.mean(boolean)
    recall = np.sum(boolean)/len(gold_cp)
    return (precision+recall+(1-dld))/3

    
    # reward = 0
    # ap=0
    # reciprocal_rank=0
    # cp = list(deepcopy(cp))
    # true =len(gold_cp)
    # size = len(cp)
    # inp = np.zeros(size)
    # for rank,val in enumerate(gold_cp) :
    #     if val  in cp:
    #         inp[cp.index(val)]=1
    # if true:
    #     ap=average_precision(inp)*(sum(inp>0)/true)
    # reciprocal_rank = mean_reciprocal_rank([inp])
    # last_reward =0 
    # i_p =cp.index(0)
    # i_g =len(gold_cp)-1
    # last_reward= 3.0/(float(abs(i_p-i_g))+3.0)
    # rewards = [(ap+last_reward)/2,dcg_at_k(inp,size)]
    # return rewards[reward_type-1]





class ContextualBandit():
    def __init__(self,b=20, rl_baseline_method="greedy",sample_method="herke",vocab= None,device='cpu'):
        self.probs_torch = None
        self.probs_numpy = None
        self.max_num_of_rec = None
        self.context = None
        self.method = sample_method
        self.device= device
        self.global_avg_reward = 0.
        self.train_examples_seen = 0.
        self.vocab = vocab
        self.rl_baseline_method = rl_baseline_method
        self.b = b  # batch_size for calculating the gradient

    def sample(self,prob,context,max_num_of_rec=10,reward_type=1,prt=False):
        """
        :return: samples b number of content plans
        """
        self.update_data_instance(prob, context, max_num_of_rec)
        self.train_examples_seen += 1
        batch_index_and_loss_lists = self.sample_batch(self.b)  
        greedy_index_list, _ = self.generate_index_list_and_loss("greedy") 
        return batch_index_and_loss_lists, greedy_index_list
        
    # def calculate_loss(self,gold_summary,summaries,gold_cp,cp,reward_type='default'):
        
    #     gold_summary = self.generate_summary(gold_summary)
    #     for i in range(len(summaries)):
    #         summaries[i][0] = self.generate_summary(summaries[i][0])
    #     greedy_summary = summaries.pop(-1)
    #     batch_rewards = [
    #         generate_reward(gold_summary,summary[0],gold_cp,cp,reward_type)
    #         for summary in summaries
    #     ]
    #     greedy_reward = generate_reward(gold_summary,greedy_summary[0],gold_cp,cp,reward_type)
    #     rl_baseline_reward = self.compute_baseline(batch_rewards,greedy_reward)
    #     loss = self.generate_batch_loss(summaries, batch_rewards, rl_baseline_reward)
        
    #     return loss, greedy_reward
    def calculate_loss(self,sample_content,gold_cp,greedy_cp,reward_type=1):
        
        # gold_summary = self.generate_summary(gold_summary)
        # for i in range(len(summaries)):
        #     summaries[i][0] = self.generate_summary(summaries[i][0])
        # greedy_summary = summaries.pop(-1)
        batch_rewards = [
            generate_reward(None,None,gold_cp,cp[0],reward_type)
            for cp in sample_content
        ]
        greedy_reward = generate_reward(None,None,gold_cp,greedy_cp,reward_type)
        rl_baseline_reward = self.compute_baseline(batch_rewards,greedy_reward)
        loss = self.generate_batch_loss(sample_content, batch_rewards, rl_baseline_reward)
        
        return loss, greedy_reward

    # def validate(self, probs, context, max_num_of_rec=3):
    #     """
    #     :return: validation_loss_of_the current example
    #     """
    #     self.update_data_instance(probs, context, max_num_of_rec)
    #     record_index_list, _ = self.generate_index_list_and_loss("greedy")
    #     reward_tuple = generate_reward(self.context.labels,record_index_list)
    #     return reward_tuple

    def update_data_instance(self, probs, context, max_num_of_rec=3):
        # self.probs_torch = probs
        # self.probs_torch = torch.clamp(probs, 1e-6, 1 - 1e-6)  # this just make sure no zero
        self.probs_torch = probs.clone() * 0.9999 + 0.00005  # this just make sure no zero
        probs_numpy = probs.data.cpu().numpy()
        self.probs_numpy = np.reshape(probs_numpy, len(probs_numpy))
        self.context = context
        self.max_num_of_rec = min(len(probs_numpy), max_num_of_rec)

    def generate_index_list_and_loss(self, sample_method="sample"):
        """
        :param sample_method: "leadk,sample, greedy"
        :return: return a list of record indexes for next step of computation
        """
        if sample_method == "lead_oracle":
            return range(self.max_num_of_rec), 0
        else:  # either "sample" or "greedy" based on the prob_list
            return return_record_index(self.probs_numpy, self.probs_torch,
                                        sample_method=sample_method, max_num_of_rec=self.max_num_of_rec,method=self.method,device=self.device)

    def generate_summary(self, summary):
        return [self.vocab.index_to_word[i.data] for i in summary]

    def sample_batch(self, b):
        batch_index_and_loss_lists = [self.generate_index_list_and_loss() for i in range(b)]
        return batch_index_and_loss_lists

    def compute_baseline(self, batch_rewards,greedy_reward):
        def running_avg(t, old_mean, new_score):
            return (t - 1) / t * old_mean + new_score / t

        batch_avg_reward = np.mean(batch_rewards)
        batch_median_reward = np.median(batch_rewards)
        self.global_avg_reward = running_avg(self.train_examples_seen, self.global_avg_reward, batch_avg_reward)

        if self.rl_baseline_method == "batch_avg":
            return batch_avg_reward
        if self.rl_baseline_method == "batch_med":
            return batch_median_reward
        elif self.rl_baseline_method == "global_avg":
            return self.global_avg_reward
        elif self.rl_baseline_method == "greedy":
            return greedy_reward
        else:  # none
            return 0

    def generate_batch_loss(self, batch_index_and_loss_lists, batch_rewards, rl_baseline_reward):
        loss_list = [
            batch_index_and_loss_lists[i][1] * ((rl_baseline_reward - batch_rewards[i]) / (rl_baseline_reward + 1e-9))
            for i in range(len(batch_rewards))
        ]
        avg_loss = sum(loss_list) / (float(len(loss_list)) + 1e-8)
        return avg_loss

 


