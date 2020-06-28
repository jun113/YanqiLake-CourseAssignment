import gym
import numpy as np
#from IPython.display import clear_output
import os
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main():
    env = gym.make('FrozenLake-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
def test():
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
def Q_learning_FL():
    env=gym.make('FrozenLake-v0').env

    env.render()

    print("Action Space {}".format(env.action_space))
    print("State Space {}".format(env.observation_space))
    q_table = np.zeros([env.observation_space.n,env.action_space.n])

    """Training the agent"""

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    for i in range(1, 101):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            os.system('cls')
            print('epoch: %d'%(i))    
            env.render()
            next_state, reward, done, info = env.step(action) 
        
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
        
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        
        if i % 100 == 0:
            os.system('cls')
        #    clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished.\n")

    """Evaluate agent's performance after Q-learning"""

    total_epochs, total_penalties = 0, 0
    episodes = 100

    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
    
        done = False
    
        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")

def RL_FL(alg):

    # gym创建冰湖环境
    env = gym.make('FrozenLake-v0')
    env.render()  # 显示初始environment
    # 初始化Q表格，矩阵维度为【S,A】，即状态数*动作数
    Q_all = np.zeros([env.observation_space.n, env.action_space.n])
    # 设置参数,
    # 其中α\alpha 为学习速率（learning rate），γ\gamma为折扣因子（discount factor）
    alpha = 0.8
    gamma = 0.95
    num_episodes = 350
    #
    Alpha = np.arange(0.75, 1, 0.02)
    Gamma = np.arange(0.1, 1, 0.05)
    #Alpha = np.ones_like(Gamma)*0.97
    # Training
    correct_train = np.zeros([len(Alpha), len(Gamma)])
    correct_test = np.zeros([len(Alpha), len(Gamma)])
    for k in range(len(Alpha)):
        for p in range(len(Gamma)):
            Q_all = np.zeros([env.observation_space.n, env.action_space.n])
            alpha = Alpha[k]
            gamma = Gamma[p]

            # training
            rList = []
            for i in range(num_episodes):
                # 初始化环境，并开始观察

                s = env.reset()
                rAll = 0
                d = False
                j = 0
                # 最大步数
                while j < 99:
                    j += 1
                    # 贪婪动作选择，含嗓声干扰
                    '''
                    print(Q_all[s,:])
                    print(np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
                    print(Q_all[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
                    print('-------------')
                    '''
                    a = np.argmax(Q_all[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))

                    # 从环境中得到新的状态和回报
                    s1, r, d, _ = env.step(a)
                    # 更新Q表
                    if alg == 'Q-learning':
                        # Q-learning
                        Q_all[s, a] = Q_all[s, a] + alpha * (r + gamma * np.max(Q_all[s1, :]) - Q_all[s, a])
                        # sarsa
                    elif alg == 'sarsa':
                        a_ = np.argmax(Q_all[s1, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
                        Q_all[s, a] = Q_all[s, a] + alpha * (r + gamma * Q_all[s1, a_] - Q_all[s, a])
                    # 累加回报
                    rAll += r
                    # 更新状态
                    s = s1
                    # Game Over
                    if d:
                        break
                rList.append(rAll)
            correct_train[k, p] = (sum(rList) / num_episodes)
            # test
            rList = []
            for i in range(num_episodes):
                # 初始化环境，并开始观察
                s = env.reset()
                rAll = 0
                d = False
                j = 0
                # 最大步数
                while j < 99:
                    # 贪婪动作选择，含嗓声干扰
                    a = np.argmax(Q_all[s, :])
                    # 从环境中得到新的状态和回报
                    s1, r, d, _ = env.step(a)
                    # # 更新Q表
                    # Q_all[s, a] = Q_all[s, a] + alpha * (r + gamma * np.max(Q_all[s1, :]) - Q_all[s, a])
                    # 累加回报
                    rAll += r
                    # 更新状态
                    s = s1
                    # Game Over
                    if d:
                        break
                rList.append(rAll)
            correct_test[k, p] = sum(rList) / num_episodes

            print("Score over time：" + str(sum(rList) / num_episodes))
            if alg == 'Q-learning':
                # Q-learning
                print('Q-learning: Alpha=%f\tGamma=%f'%(alpha,gamma))
                print( Q_all)
            elif alg == 'sarsa':
                # sarsa
                print('sarsa: Alpha=%f\tGamma=%f'%(alpha,gamma))
                print( Q_all)

    # Test
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(correct_test, interpolation='nearest', cmap='rainbow',
               extent=[0.75, 1, 0, 1],
               origin='lower', aspect='auto')
    if alg == 'Q-learning':
        # Q-learning
        plt.title('Q-learning Accuracy')
        # sarsa
    elif alg == 'sarsa':
        plt.title('sarsa Accuracy')
    
    plt.ylabel('Gamma')
    plt.xlabel('Alpha')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    plt.show()
    return correct_test
if __name__ == "__main__":
#    main()
#    test()
#    Q_learning_FL()

    correct_ql=RL_FL('Q-learning')
    correct_sarsa=RL_FL('sarsa')
    temp=correct_ql-correct_sarsa

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(temp, interpolation='nearest', cmap='rainbow',
               extent=[0.75, 1, 0, 1],
               origin='lower', aspect='auto')
    
    plt.ylabel('Gamma')
    plt.xlabel('Alpha')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    plt.show()