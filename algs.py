from distribution import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict


class EpsGreedy:
    def __init__(self, eps, offer_cont):
        self.eps = eps
        self.offer_cont = offer_cont
        self.K = len(self.offer_cont)
        self.reward_sum = np.zeros(self.K)

    def get_action(self, cust_context=None):
        candidates = np.argwhere(self.reward_sum >= \
            self.reward_sum.max()).flatten()
        idx = int(np.random.choice(candidates.size, 1, replace=False))
        return candidates[idx] if np.random.uniform() >= self.eps \
            else np.random.randint(0, self.K)

    def set_reward(self, action, reward):
        self.reward_sum[action] += reward

class UCB1:
    def __init__(self, offer_cont):
        self.offer_cont = offer_cont
        self.K = len(self.offer_cont)
        self.reward_sum = np.zeros(self.K)
        self.action_choice = np.zeros(self.K)
        self.round_num = 0
        # for logging
        self.log_upper_bound = defaultdict(list)
        self.log_mean_reward = defaultdict(list)
        self.log_uncer = defaultdict(list)
        self.log_reward_sum = defaultdict(list)
    
    def get_action(self, cust_context=None):
        if self.round_num < self.K:
            ub = np.zeros(self.K)
            mean_reward = np.zeros(self.K)
            uncertainty = np.zeros(self.K)
            best_action = self.round_num
        else:
            mean_reward = self.reward_sum / (self.round_num + 1)
            uncertainty = np.sqrt(2*np.log(self.round_num) / self.action_choice)
            ub = mean_reward + uncertainty
            # select the random maximal action (if more than one)
            candidates = np.argwhere(ub >= ub.max()).flatten()
            idx = int(np.random.choice(len(candidates), 1, replace=False))
            best_action = candidates[idx]

        self.action_choice[best_action] += 1
        self.round_num += 1

        # log upper bound, uncertainty and reward
        for k in range(self.K):
            self.log_upper_bound[k].append(ub[k])
            self.log_reward_sum[k].append(self.reward_sum[k])
            self.log_mean_reward[k].append(mean_reward[k])
            self.log_uncer[k].append(uncertainty[k])

        return best_action

    def set_reward(self, action, reward):
        self.reward_sum[action] += reward

    def show_log(self):
        fig, ax = plt.subplots(1,3, figsize=(15,5))
        for k in range(self.K):
            ax[0].plot(
                range(1, self.round_num+1), \
                self.log_mean_reward[k], 
                label=str(self.offer_cont[k])
            )
            ax[1].plot(
                range(1, self.round_num+1), \
                self.log_uncer[k], 
                label=str(self.offer_cont[k])
            )
            ax[2].plot(
                range(1, self.round_num+1),
                self.log_reward_sum[k],
                label=str(self.offer_cont[k])
            )
            ax[2].text(
                self.round_num, 
                self.log_reward_sum[k][-1] + 5, 
                str(self.log_reward_sum[k][-1])
            )
        ax[0].set_ylabel('Mean reward')
        ax[0].set_xlabel('Iteration')
        ax[0].legend()
        ax[1].set_ylabel('Uncertainty')
        ax[1].set_xlabel('Iteration')
        ax[1].legend()
        ax[2].set_ylabel('Cummulative reward')
        ax[2].set_xlabel('Iteration')
        ax[2].legend()
        fig.suptitle('UCB', fontsize=16)
        fig.tight_layout()
        plt.show()


class LinUCB:
    """Implementation of Linear Upper Confidence Bound (LinUCB)
    Ref: https://arxiv.org/abs/1003.0146
    """
    def __init__(self, offer_cont):
        self.offer_cont = offer_cont
        self.K = len(self.offer_cont)
        self.reward_sum = np.zeros(self.K)
        self.action_choice = np.zeros(self.K)
        self.round_num = 0
        # parameters
        self.theta = {}
        self.b = {}
        self.curr_context = {}
        self.A_inv = {}
        # for logging
        self.log_upper_bound = defaultdict(list)
        self.log_pred_mean = defaultdict(list)
        self.log_pred_var = defaultdict(list)
        self.log_reward_sum = defaultdict(list)
        

    def get_action(self, cust_context):
        ub, pred_mean, pred_var = {}, {}, {}
        alpha = 1 + 1/(self.round_num + 1)

        # loop over K arms
        best_action = None
        for k in range(self.K):
            self.curr_context[k] = np.expand_dims(
                np.r_[cust_context.to_numpy(), self.offer_cont[k].to_numpy()], 
                axis=1
            )
            d = self.curr_context[k].size
            if k not in self.A_inv:
                self.b[k] = np.zeros((d,1))
                self.A_inv[k] = np.eye(d)

            self.theta[k] = self.A_inv[k] @ self.b[k]
            pred_var[k] = self.curr_context[k].T @ self.A_inv[k] @ self.curr_context[k]
            pred_mean[k] = self.theta[k].T @ self.curr_context[k]
            ub[k] = pred_mean[k] + alpha * np.sqrt(pred_var[k])
        
        # select the random maximal action (if more than one)
        max_ub = max(ub.values())
        candidates = [k for k in ub.keys() if ub[k] >= max_ub]
        idx = int(np.random.choice(len(candidates), 1, replace=False))
        best_action = candidates[idx]

        self.action_choice[best_action] += 1
        self.round_num += 1

        # log upper bound, predictive variance and reward
        for k in range(self.K):
            self.log_upper_bound[k].append(float(ub[k]))
            self.log_pred_mean[k].append(float(pred_mean[k]))
            self.log_pred_var[k].append(float(pred_var[k]))
            self.log_reward_sum[k].append(self.reward_sum[k])

        return best_action
        
    def set_reward(self, action, reward):
        self.reward_sum[action] += reward
        
        # update parameters
        x_a = self.curr_context[action]
        #self.A[action] += x_a @ x_a.T
        self.b[action] += reward * self.curr_context[action]

        # avoid inversion using Sherman-Morrison
        self.A_inv[action] = self.A_inv[action] - (self.A_inv[action] @ x_a) @ (x_a.T @ self.A_inv[action]) / \
            (1 + x_a.T @ self.A_inv[action] @ x_a)
        
    
    def show_log(self):
        fig, ax = plt.subplots(1,3, figsize=(15,5))
        for k in range(self.K):
            ax[0].plot(
                range(1, self.round_num+1), \
                self.log_pred_mean[k], 
                label=str(self.offer_cont[k])
            )
            ax[1].plot(
                range(1, self.round_num+1), \
                self.log_pred_var[k], 
                label=str(self.offer_cont[k])
            )
            ax[2].plot(
                range(1, self.round_num+1),
                self.log_reward_sum[k],
                label=str(self.offer_cont[k])
            )
            ax[2].text(
                self.round_num, 
                self.log_reward_sum[k][-1] + 5, 
                str(self.log_reward_sum[k][-1])
            )
        ax[0].set_ylabel('Predictive mean')
        ax[0].set_xlabel('Iteration')
        ax[0].legend()
        ax[1].set_ylabel('Predictive variance')
        ax[1].set_xlabel('Iteration')
        ax[1].legend()
        ax[2].set_ylabel('Cummulative reward')
        ax[2].set_xlabel('Iteration')
        ax[2].legend()
        fig.suptitle('LinUCB', fontsize=16)
        fig.tight_layout()
        plt.show()