import numpy as np
from distribution import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from algs import (
    EpsGreedy,
    UCB1,
    LinUCB
)

class Simulator:
    def __init__(self, learner):
        self.learner = learner
    
    def simulate(self, 
                 cust_cont_cls: CustomerContext,
                 buy_behavior_cls: BuyBehavior,
                 T=1000,
                 show_log=False):
        
        offer_cont = self.learner.offer_cont
        rewards = []
        if show_log:
            print('{:<10s}{:<30s}{:<40s}{:<30s}'.format('iter', 'user_context', 'best_offer', 'reward'))
        
        for t in range(T):
            # get customer context
            cust_cont = cust_cont_cls.sample()
            # get action
            action = self.learner.get_action(cust_cont)
            # set reward
            reward = buy_behavior_cls.sample(cust_cont, offer_cont[action])
            self.learner.set_reward(action, reward)
            rewards.append(reward)
            # logging
            if show_log:
                print('{:<10d}{:<30s}{:<40s}{:<30.1f}'.format(
                    t+1,
                    str(cust_cont),
                    str(offer_cont[action]),
                    round(reward)
                ))
        return np.cumsum(rewards)


if __name__ == "__main__":
    # number of rounds
    T = 1000

    # offer settings
    offer_cont = [
        OfferContext(best_price=0, best_pdd=1),
        OfferContext(best_price=0, best_pdd=0),
        OfferContext(best_price=1, best_pdd=0)
    ]
    
    # simulation
    epsgreedy = EpsGreedy(eps=0.1, offer_cont=offer_cont)
    sim = Simulator(epsgreedy)
    cum_reward = sim.simulate(CustomerContext, BuyBehavior, T=T)
    plt.plot(range(1, len(cum_reward)+1), cum_reward, label='0.1-Greedy')

    random = EpsGreedy(eps=0.5, offer_cont=offer_cont)
    sim = Simulator(random)
    cum_reward = sim.simulate(CustomerContext, BuyBehavior, T=T)
    plt.plot(range(1, len(cum_reward)+1), cum_reward, label='Random')

    ucb1 = UCB1(offer_cont=offer_cont)
    sim = Simulator(ucb1)
    cum_reward = sim.simulate(CustomerContext, BuyBehavior, T=T)
    plt.plot(range(1, len(cum_reward)+1), cum_reward, label='UCB1')

    linucb = LinUCB(offer_cont=offer_cont)
    sim = Simulator(linucb)
    cum_reward = sim.simulate(CustomerContext, BuyBehavior, T=T, show_log=True)
    plt.plot(range(1, len(cum_reward)+1), cum_reward, label='LinUCB (contextual)')

    plt.xlabel('Iteration')
    plt.ylabel('Cummulative reward')

    plt.legend()
    plt.show()

    # show details of UCB and LinUCB
    ucb1.show_log()
    linucb.show_log()