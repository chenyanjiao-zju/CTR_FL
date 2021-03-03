import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == '__main__':
    sns.set()
    neg = np.load("neg.npy")
    pos = np.load("pos.npy")
    data = np.load("pred.npy")
    poi = np.load("attack.npy")

    cd = np.where(data > 0.9)

    # print(cd)
    print(neg.shape[0] / pos.shape[0])
    # print(neg.shape[0]+pos.shape[0])
    sns.set(style="white", palette="muted", color_codes=True)
    sns.distplot(data[neg], kde=False, fit=stats.gamma, fit_kws={"color": "black"}, label='0', hist=False)
    sns.distplot(data[pos], kde=False, fit=stats.gamma, fit_kws={"color": "r"}, label='1', hist=False)

    # sns.distplot(poi[neg], kde=False, fit=stats.chi2, fit_kws={"color": "b"}, label='0_p', hist=False)
    # sns.distplot(poi[pos], kde=False, fit=stats.gamma, fit_kws={"color": "y"}, label='1_p', hist=False)

    plt.legend()
    plt.show()