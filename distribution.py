import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class CustomerContext:
    """Class defining user context"""
    def __init__(self, cheap, fast):
        self.cheap = cheap 
        self.fast = fast

    @classmethod
    def sample(cls):
        """Class method for generating stochastic user context"""
        cheap = np.clip(0.1*np.random.randn() + 0.7, 0, 1)
        fast  = np.clip(0.1*np.random.randn() + 0.6, 0, 1)
        return cls(cheap, fast)
    
    def to_numpy(self):
        """Get array representation"""
        return np.array([self.cheap, self.fast])

    def __str__(self):
        return str('cheap: %.2f, fast: %.2f' % (self.cheap, self.fast))

class OfferContext:
    """Class defining offer context"""
    def __init__(self, best_price, best_pdd):
        self.best_price = best_price 
        self.best_pdd = best_pdd
    
    @classmethod
    def sample(cls):
        """Class method for generating stochastic offer context"""
        return cls(OfferContext(int(np.random.uniform() >= 0.5), int(np.random.uniform() >= 0.5)))
    
    def to_numpy(self):
        """Get array representation"""
        return np.array([self.best_price, self.best_pdd])

    def __str__(self):
        return str('best_price: %.2f, best_pdd: %.2f' % (self.best_price, self.best_pdd))

class BuyBehavior:
    """Class defining buying behavior based on user and offer context"""
    @classmethod
    def sample(cls, cust_cont, offer_cont):
        """Return stochastic value: 1 (buy) and 0 (do not buy)"""
        return  int(np.random.uniform() < 
                0.5*(cust_cont.cheap * offer_cont.best_price + cust_cont.fast * offer_cont.best_pdd))
    

if __name__ == '__main__':
    cust_cont = []
    for _ in range(5000):
        cust_cont.append(CustomerContext.sample())

    offer_cont = [
        OfferContext(0, 0),
        OfferContext(0, 1),
        OfferContext(1, 0)
    ]
    
    fig, ax = plt.subplots(1, 3, figsize=(10,5))
    ax[0].hist([c.cheap for c in cust_cont])
    ax[0].set_xlabel('Prefering cheap')
    ax[0].set_ylabel('Number of customers')

    ax[1].hist([c.fast for c in cust_cont])
    ax[1].set_xlabel('Prefering fast')
    ax[1].set_ylabel('Number of customers')

    ax[2].scatter(\
        [o.best_price for o in offer_cont], 
        [o.best_pdd for o in offer_cont], 
    )
    ax[2].set_xlabel('Best price')
    ax[2].set_ylabel('Best pdd')
    ax[2].set_xticks([0, 1])
    ax[2].set_yticks([0, 1])

    fig.tight_layout()
    plt.show()