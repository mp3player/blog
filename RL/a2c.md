# preface 
+ 先来说说我个人对于强化学习(Reinforcement Learning, RL)的理解，不是关于算法层次的，而是基于他的工作原理的。我们目前都可以看到AI的应用十分广泛，而且经过这些年的发展，各种AI的能力也是特别强的。如果我们从一个旁观者的角度来看一个语言模型的话，我们可能会这样设计它。首先，它应该是一个神经网络；然后他可以对我的输入进行反馈。然后我们来考虑他的输出和我们的输入应该是相容的，也就是说至少是正确的，也是最基本的要求。然后我们可能会想到，同一个回答，其实有很多种的组织方式，但是根据supervised learning的方法的话，我们的输入其实只有一个对应的label。这样的话其实是没办法让模型产生多样性的反馈的。即使我们更具体一点，为同一个input准备多个label，这样看似能够有多样性，但是理论上的输出应该是这些label的mean。这样分析的话，其实也很难实现一个类人的语言模型。
+ 为了解决这个问题，就需要RL的参与，详细的内容可以参考InstructGPT的内容，这里直接给出我觉得通俗易懂的理解。RL其实就是根据情况来做决策的，这一特点使得他的工作方式看起来像人一样。这一点表现在语言模型中其实就是他的输出看起来是有情感的。有时候大家初学RL的时候有可能被它里边的一些术语给搞得稀里糊涂的，下面我为了让人理解，可能会用到生活中的某些情况来类比。


# 强化学习介绍

+ 这里就不用介绍什么具体的定义了，做工程的几乎不会去在乎这些内容。那么这里就从我们经常在书上看到的一些内容解释开始。
+ 第一个就是马尔科夫决策过程(Markov Decision Process, MDP )，这是几乎所有的RL课程都会提到的一个内容。他对于算法的理解可能并没有什么作用，这里简单解释一下。其实核心就是Markov性质，意味着他只基于上一次的内容做决策，简单来说就是他的记忆只停留在上一次，并且基于它做决策。
+ 第二个就是RL里边经常提到的一些术语，比如Environment、Policy、Agent等等。这里类比一下，Environment就是我们的问题，我们可以从中知道目前问题的状态；然后Agent通常被称为智能体，你可以类比为一个没有主见的人；他必须搭配Policy帮助他对某些情况做决定。
+ 第三个就是RL的算法，我目前了解到的其实分为两个类别，Policy-based和Value-based。这是我认为的，但是你可能在一些文献中看到他有更多的划分方式。但在我看来这两种方式其实就把他们的本质区分开了。至于为什么这么划分，光凭文字叙述有点困难，需要一些实际的场景来帮助理解。

# 强化学习的工作方式

+ 我们介绍了RL中包含的一些内容，这里来看一下他的工作方式。一种是从直观来理解，而另一种则是从训练方法来理解的。
+ 直观理解就是我们为了解决Environment的问题，需要观察他的状态(states)，然后根据Policy的决策，让一个没有主见的Agent做出相应的Action，从而逼近Environment问题的解，然后循环这一过程就行了。但是仅仅重复这一步骤是不会有任何进展的，因为我们不知道问题被解决的情况，所以我们还需要另一个反馈，就是reward，表明我们对问题的解决发展得怎么样。reward高就意味着好，低就意味着差。
+ 另一种就是从训练的角度来看，他其实就是一个supervised learning。我们需要Policy(Neural Network)在看到Environment的States(Input)后做决策(Output)，那么就需要准备(States , Output)，来训练我们的Policy，让他能够正确得解决问题，而这个监督信号，其实就来自于Reward。

# 第一个例子，CartPole-v1
+ 其实到这里，就已经对RL那些抽象的概念介绍得差不多了。但是没有例子的支撑，上面的内容看起来仍然很抽象。所以这里我们用一个具体的例子来介绍。就是gymnasium的CartPole-v1(动手能力强的可以试一下，这里只是举一个例子，下一次我们将更加详细得介绍他们的使用)。这个问题十分简单，有一根竖着的竹竿，你需要左右运动，以确保这个竹竿不会倒下。而他的reward其实就是这个竹竿竖着的时间。那我们的Env或者说问题就是这个CartPole-v1，他的状态就是这个竹竿的位置等信息，然后我们的目标就是让这个竹竿竖着的时间尽可能长。Policy输入这个竹竿的位置等信息，输出一个Action的概率分布；之后Agent则根据这个概率分布，选择一个Action，即往左还是往右移动。

+ 如果你去到gymnasium的官网的话，你可以看到一个简答的例子，但是要想运行这个例子，你可能得花点时间解决一下某些pip包依赖的问题

```python
import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()

```

+ 所以你可以将他替换成Carpole-v1，就可以直接运行

```python
# Initialise the environment
env = gym.make("CartPole-v1", render_mode="human")

# !!!!!!!!!!! you should init you policy 
# Policy = torch.nn.Module( Env.state.shape )

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy

    # Probability Ditribution from Policy
    # dist = Policy( observation )

    # replace this whth 
    # action = dist.sample()
    action = env.action_space.sample()
    


    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # signal = Learning_Signal( reward )
    # update( Policy , signal )

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

+ 上面我写了一些伪代码注释，对RL所需的一些内容做了补充，可以作为一个最开始的理解，首先我们会看到Env的observation，然后输入到Policy中，之后产生一个dist分布，并从里边采样到一个action，之后通过env.step( action ) do Action，然后会得到reward。最后就是根据Reward产生的更新信号，来更新policy。接下来重复这个步骤。
+ 在上面的内容中，你可能好奇没有看到Agent出现。其实这里边很多东西都是抽象的，agent的功能其实就是观察Env的states和做Action。被包含在了Env中。
+ 这只是一个简单的例子，可能看起来还是比较抽象，不过别急。当我们将这个用具体的方法来求解的时候，就会变得清晰了。