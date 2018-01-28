import matplotlib.pyplot as plt
import numpy as np


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


if __name__ == "__main__":
    train_rewards_list = np.load("nips_train_rewards.npy")
    test_rewards_list = np.load("nips_test_rewards.npy")

    eps, rews = np.array(train_rewards_list).T
    test_rews = np.array(test_rewards_list)
    smoothed_rews = running_mean(rews, 100)
    test_smoothed_rews = running_mean(test_rews, 100)
    titile = "NIPS_DQN"
    plt.plot(eps[-len(smoothed_rews):], smoothed_rews,
             label="last100_mean_train")
    plt.plot(eps, rews, color='grey', alpha=0.3)
    plt.plot(eps[-len(test_smoothed_rews):],
             test_smoothed_rews, label='last100_mean_test')
    plt.plot(test_rews, color='orange', alpha=0.3)
    plt.title("CartPole-v1 "+titile)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig(titile+".png", format='png', dpi=300)
    plt.show()
