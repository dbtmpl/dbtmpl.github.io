---
layout: post
title: Exploring Variance in N-Step Actor-Critic Methods
description: Comparing n-step bootstrapping returns, from TD(0) to Monte Carlo, in actor-critic methods across several Gym environments.
kicker: Student project
authors: >-
  <strong>David Biertimpel</strong>, <a href="https://github.com/ClaartjeBarkhof">Claartje Barkhof</a>,
  <a href="https://github.com/victoriafoing">Vicky Foing</a> &amp; <a href="https://github.com/axlindt">Alex Lindt</a>
affiliation: University of Amsterdam
date: 2019-10-01
code: https://github.com/axlindt/variance_n_step_actor_critic
permalink: /rl_section.html
math: true
back_url: /#projects
back_label: Back to projects
---

<figure>
  <img src="{{ '/assets/rl_article/lunar_lander.gif' | relative_url }}" alt="Agents trained with different methods landing in the Lunar Lander environment">
  <figcaption><b>Figure 1.</b> Behavior of our agents in the Lunar Lander environment.</figcaption>
</figure>

## Introduction

Imagine two athletes training for a competition. The first athlete's coach is a master of his trade but, due to his advanced age, teaches techniques that are consistent but slightly antiquated. The second athlete is coached by a rising star who just graduated from the academy. Although her instructions fit the modern sport exactly, she is still inconsistent in her decisions because of her lack of experience. Which athlete will perform better in the competition? This question reflects the bias-variance trade-off in reinforcement learning. While the first athlete exactly learns the *biased* strategy of his coach, the instructions the second athlete receives have high *variance*.

In this blog article, we investigate bias and variance in actor-critic methods by comparing different n-step bootstrapping returns ranging from TD(0) to Monte Carlo across multiple environments. The core of actor-critic methods is the policy gradient, which recently achieved great success in various control tasks such as 3D locomotion<sup class="cite"><a href="#ref-1">[1]</a></sup>, playing video games<sup class="cite"><a href="#ref-2">[2]</a></sup> or solving a Rubik's cube<sup class="cite"><a href="#ref-3">[3]</a></sup>. The biggest advantages of policy gradient methods are that they are well suited to continuous action spaces, implicitly learn stochastic policies, and have good convergence properties overall. However, policy gradient methods have also proven to show high variance in many aspects, such as performance over multiple runs (different random seeds), sensitivity to small changes in hyperparameters, and low robustness to unexpected changes in the environment<sup class="cite"><a href="#ref-4">[4]</a>, <a href="#ref-5">[5]</a></sup>. This makes policy gradient methods not always reliable and complicates the reproduction of results. The most obvious source of variance is the variance in sampled trajectories and therefore in the accumulated rewards (the return) the agent receives along these trajectories. We have a strong incentive to reduce this variance, as it hinders accurate gradient estimation and consequently slows learning. This is why, over the next sections, we investigate various n-step estimates of the return, ranging from *high bias* and *low variance* TD(0) to *low bias* and *high variance* Monte Carlo.

### Policy Gradient Methods

Before we dive into the topic, let's refresh our knowledge of reinforcement learning and policy gradients. In reinforcement learning, our goal is to maximize the expected return $$G$$ that the agent receives by interacting with the environment under a policy $$\pi$$:

$$
J = \mathbb{E}_{\tau}\big[ G(\tau) \big]
$$

In policy gradient methods, we directly model the policy with a parameterized (often nonlinear) function $$\pi_\theta$$, such as a neural network, and perform gradient updates in the direction of maximal return. To do this, we take the gradient of the expected return using the log-derivative identity:

$$
\begin{aligned}
\nabla_\theta J &= \nabla_\theta\, \mathbb{E}_{\tau}\big[ G(\tau) \big] \\
&= \mathbb{E}_{\tau}\big[ \nabla_\theta \log p_\theta(\tau)\, G(\tau) \big] \\
&= \mathbb{E}_{\tau}\Big[ G(\tau)\, \nabla_\theta \sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t) \Big] \\
&= \mathbb{E}_{\tau}\Big[ G(\tau) \sum_{t=0}^{T} \frac{\nabla_\theta \pi_\theta(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)} \Big]
\end{aligned}
$$

The third line holds because the environment's transition probabilities in $$p_\theta(\tau)$$ do not depend on $$\theta$$, so only the policy terms survive the gradient.

### REINFORCE

This results in the most basic policy gradient method, called REINFORCE. The intuition is that we make actions associated with a high return more likely. The last line of the derivation also shows explicitly that we correct for the frequency with which the policy selects each action.

Without getting lost in further derivations, we can easily imagine that the return of the full trajectory $$G(\tau)$$ has high variance between episodes, due to the stochasticity of both the environment's dynamics and the policy. So we might want to use different estimates of the return. One prominent way to reduce variance is to subtract a baseline from the return, which is called REINFORCE with baseline. A good candidate for the baseline is the *state-value* function $$V_\pi(S)$$, which describes how much reward the agent can expect to receive from state $$S$$ onward when following policy $$\pi$$. As with the policy, we can estimate $$V_\pi(S)$$ with a nonlinear function approximator such as a neural network. The intuition behind subtracting $$V_\pi(S)$$ from the actual return is that we separate the quality of the action from the quality of the current state. Note that a state-dependent baseline does not bias the gradient; it only reduces its variance.

Instead of using the full return of the trajectory, we can also calculate the TD target

$$
R_{t+1} + \gamma V_\pi(S_{t+1}),
$$

which becomes the TD error when combined with the baseline $$V_\pi(S_t)$$. This reduces variance to a minimum, as we only look at the next step the agent takes. In return it introduces bias, since $$V_\pi$$ is only an estimate: it is randomly initialized and does not reflect the true *state-value* at the beginning of training. The intuition for this return estimate is that we are *happily surprised* if the TD error is positive, since we received more reward than expected, and thus increase the probability of the corresponding action. Conversely, if the TD error is negative, we received less reward than expected and disappointedly lower the probability of the corresponding action.

A different TD error can be built from the *action-value* function $$Q_\pi(S, A)$$. The action-value function describes the future reward the agent can expect when taking action $$A$$ in state $$S$$ and following policy $$\pi$$ afterwards. The corresponding TD error is

$$
R_{t+1} + \gamma \max_{a} Q_\pi(S_{t+1}, a) - Q_\pi(S_t, A_t).
$$

For the same reasons as $$V_\pi$$, the bootstrapped $$Q_\pi$$ is also a biased estimate of the action-value function.

### Actor-Critic Methods

By using the TD error instead of the full return, we finally arrive at the methods this article focuses on: actor-critic methods. In actor-critic methods, we use the estimates of $$V_\pi$$ or $$Q_\pi$$ not only as a baseline but also for bootstrapping, that is, we update an estimate of $$V_\pi$$ or $$Q_\pi$$ with another estimate of $$V_\pi$$ or $$Q_\pi$$. Bootstrapping introduces significant bias, which in turn reduces the variance of the updates and should therefore stabilize and accelerate learning. Actor-critic methods get their name from combining a parameterized policy $$\pi_\theta$$ that decides which actions to take (the actor) with an estimate of $$V_\pi$$ or $$Q_\pi$$ that judges the quality of the current actions and states (the critic). So the actor tells us *how to act* and the critic tells us *how well we are acting*. As indicated above, we focus on the version where the TD error is used as the critic's estimate. This version is called advantage actor-critic, because it measures the advantage of taking an action in the current state compared to the expected return of that state.

### Policy Gradient Theorem

Perhaps you are wondering how on earth we can seamlessly swap $$G$$ for all these different return estimates. This is possible because of the policy gradient theorem, which states that for any differentiable policy $$\pi_\theta$$ and for any policy objective function $$J$$, the policy gradient is

$$
g = \mathbb{E}\Big[ \sum_{t=0}^{\infty} \Psi_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t) \Big]
  = \mathbb{E}\Big[ \sum_{t=0}^{\infty} \Psi_t\, \frac{\nabla_\theta \pi_\theta(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)} \Big],
$$

where $$\Psi_t$$ is a placeholder for different return estimates. The derivation of this theorem is quite straightforward and is nicely explained in Schulman's Generalized Advantage Estimation (GAE) paper<sup class="cite"><a href="#ref-1">[1]</a></sup> or, for people who prefer visual input, in [Hado van Hasselt's DeepMind lecture on policy gradients and actor-critics at UCL](https://www.youtube.com/watch?v=bRfUxQs6xIM).

### N-Step Bootstrapping

We have now seen several ways to choose either a biased estimate (the TD error) or a high-variance one (the Monte Carlo return in REINFORCE). However, the choice has been binary so far. To study the trade-off between bias and variance, it would be nice to have estimates between both extremes. This is where n-step returns enter the ring. Instead of bootstrapping with the estimated state-value of the next state, as in the TD error above, we can observe a longer chain of rewards before plugging in the estimated state-value. The state-value we plug in can be $$n$$ steps away, which gives *n*-step actor-critic methods their name. N-step returns integrate naturally into the actor-critic methods described above, as we only have to change the TD target. Where we had $$R_{t+1} + \gamma V_\pi(S_{t+1})$$ before, we now plug in

$$
R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^{n} V_\pi(S_{t+n}).
$$

If $$n = \infty$$, we get exactly the full Monte Carlo return $$G$$ from REINFORCE. If $$n = 1$$, the method equals the one-step TD error described above.

Based on this theory, we hypothesize that by tuning $$n$$, ranging from one-step actor-critic (high bias) to REINFORCE with baseline (high variance), we can tune the bias-variance trade-off and pick a sweet spot for each environment. In other words, we investigate whether different values of $$n$$ are beneficial for different algorithms and environments.

## Methods

To make the comparison between methods as fair as possible, we implement them all following the same pseudocode:

```text
# INITIALIZE EXPERIMENT
actor_θ, critic_w ← parameterized functions, instantiated as neural networks
α_θ, α_w          ← step sizes for actor and critic

# EXPERIMENT LOOP
for each episode in max_episodes:

    # INITIALIZE EPISODE
    state ← start state of environment
    log_probs, entropies, states, rewards ← []

    # EPISODE LOOP
    for each step t = 0, 1, ..., T:
        policy ← actor_θ(state)
        action ← sample(policy)
        next_state, reward ← environment_step(action)
        log_probs ← log_probs + log(policy[action])
        entropies ← entropies + entropy(policy)
        states    ← states + state
        rewards   ← rewards + reward
        state     ← next_state

    # UPDATE PARAMETERS (once per episode)
    𝚿 ← 𝚿_function(states, rewards, critic_w)
    loss_actor  ← mean(-log_probs * 𝚿 - 0.01 * entropies)   # policy gradient + entropy bonus
    loss_critic ← mean(𝚿²)                                   # mean squared error
    θ ← θ - α_θ * ∇_θ loss_actor
    w ← w - α_w * ∇_w loss_critic
```

The different instantiations of $$\Psi$$ are:

| Algorithm | $$\Psi_t$$ |
| --- | --- |
| REINFORCE with baseline | $$\Psi_t \leftarrow G_t - \hat{v}(S_t, w)$$ |
| Advantage actor-critic (1-step) | $$\Psi_t \leftarrow R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)$$ |
| Advantage actor-critic (n-step) | $$\Psi_t \leftarrow R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^{n} \hat{v}(S_{t+n}, w) - \hat{v}(S_t, w)$$ |
| Q actor-critic (1-step) | $$\Psi_t \leftarrow R_{t+1} + \gamma \max_a Q(S_{t+1}, a, w) - Q(S_t, A_t, w)$$ |
| Q actor-critic (n-step) | $$\Psi_t \leftarrow R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^{n} \max_a Q(S_{t+n}, a, w) - Q(S_t, A_t, w)$$ |

In contrast to how actor-critic is described in Sutton &amp; Barto<sup class="cite"><a href="#ref-6">[6]</a></sup>, we do not update our models after every time step but do a batch update at the end of every episode. This stabilizes our gradients and matches the descriptions in related papers and professional implementations<sup class="cite"><a href="#ref-7">[7]</a></sup>. Inspired by the A2C algorithm<sup class="cite"><a href="#ref-8">[8]</a></sup>, we add an entropy term to the actor loss that reflects the entropy of the categorical distribution over actions. This term flattens the action probabilities and encourages exploration.

## Experiments

### Environments

To test our hypothesis, we select three control tasks from OpenAI Gym with different characteristics.

#### CartPole

In this game, the goal is to balance a pole on a cart. By moving left and right (a discrete action space), the agent tries to keep the pole upright. The game is over when the pole tilts more than 12 degrees from upright or when the cart moves too far off the track in either direction (2.4 units). The episode ends naturally when the pole has been balanced for 200 time steps. A reward of +1 is granted for every step the pole stays up. With its straightforward state and action spaces, CartPole promises reliable results, so it serves as a stable baseline to compare our methods.

#### Lunar Lander

In the Lunar Lander environment, the agent has to land a spaceship between two flags as smoothly as possible. If the spaceship crashes, it receives a reward of −100; if it lands smoothly, it receives +100. The agent can choose from four actions: do nothing, fire the left engine, fire the main engine, and fire the right engine. Additional rewards shape the agent's behavior: a small penalty (−0.3 per frame) for firing the main engine, a reward of +10 each time a leg touches the ground, and a terrain that changes between episodes. Lunar Lander is more complex than CartPole, both in its reward structure and because the environment changes slightly from episode to episode. It is therefore interesting to see whether our agents can adapt.

#### Mountain Car

In MountainCar-v0, the agent has to drive a small car up a hill. The landscape is part of a cosine curve whose highest point is the hill. To reach the top, the car has to gain momentum by driving up and down both slopes alternately, since its engine is not strong enough to drive up the hill directly. The action space is discrete: push left, do nothing, or push right. For every time step in which the car has not reached the goal, the agent receives a reward of −1, so it is incentivized to reach the hill as fast as possible. An interesting property of this environment is that the agent first has to reach the goal by random exploration before it receives the signal that allows it to learn the optimal policy. Because of this need for stochasticity, we expect high-variance agents to perform better.

### Experimental Set-Up and Metrics

For each environment, we run experiments with nine configurations: REINFORCE with baseline and four versions of each actor-critic algorithm ($$n = 1, 2, 4, 8$$). Each configuration is run five times with the same five random seeds. To analyze performance, we store the episode durations, cumulative rewards, actor losses and critic losses, and average them across the five runs. These metrics track how quickly and accurately each algorithm learns the reward structure of the environment, which provides insight into bias and variance.

### Architecture and Hyperparameters

All algorithms use two neural networks, one for the policy and one for the value estimate. The hyperparameters are:

| Hyperparameter | Setting |
| --- | --- |
| Learning rate | 7e-4 |
| Number of episodes | 1000 |
| Discount factor | 0.99 |
| Hidden layers | two layers of 64 units |
| N-step values | 1, 2, 4, 8 |
| Number of runs | 5 |
| Random seeds | 43, 44, 45, 46, 47 |

We select these hyperparameters based on the literature and some hyperparameter optimization of our own. To ensure fair comparisons, we keep the values the same for all configurations.

## Results

### Overall Performance

First, we look at the overall performance of our methods and compare them to our baseline algorithm, REINFORCE with baseline.

#### CartPole

<figure>
  <img src="{{ '/assets/rl_article/CP_n8.png' | relative_url }}" alt="Reward curves of all algorithms on CartPole">
  <figcaption><b>Figure 2.</b> Performance of the different algorithms on CartPole.</figcaption>
</figure>

The plots show that REINFORCE with baseline outperforms the actor-critic methods, consistently reaching rewards close to 200 (the maximum score) after about 300 episodes. For advantage actor-critic, performance seems to increase with $$n$$. Interestingly, for $$n > 1$$ the agents reach rewards similar to REINFORCE with baseline, close to 175 after 300 episodes. The agent with $$n = 1$$, however, falls out of line and stagnates after 250 episodes.

The Q actor-critic agents show a different picture. Here we cannot say that performance improves with $$n$$, as the $$n = 2$$ agent behaves best. Overall, these agents do not come close to the advantage actor-critic agents. One possible reason is that we perform a Q-learning update when bootstrapping, $$R_{t+1} + \gamma \max_a Q(S_{t+1}, a, w)$$, which introduces an off-policy element into an algorithm that is essentially on-policy.

#### Lunar Lander

<figure>
  <img src="{{ '/assets/rl_article/LL_n8.png' | relative_url }}" alt="Reward curves of all algorithms on Lunar Lander">
  <figcaption><b>Figure 3.</b> Performance of the different algorithms on Lunar Lander.</figcaption>
</figure>

On Lunar Lander, REINFORCE with baseline again outperforms most of the actor-critic methods. The advantage actor-critic methods perform similarly to REINFORCE with baseline and improve steadily throughout training. In contrast, the Q actor-critic methods barely improve over the 1000 episodes. Again, the advantage actor-critic methods improve as $$n$$ increases, while the Q actor-critic methods do not. What is exciting about Lunar Lander is that at the end of training, advantage actor-critic with the largest value, $$n = 8$$, eventually outperforms REINFORCE with baseline and reaches a reward of 0. This suggests that we found a sweet spot for this experiment, where bootstrapping is better than calculating the full return.

#### Mountain Car

<figure>
  <img src="{{ '/assets/rl_article/mountain_car.gif' | relative_url }}" alt="A car rolling back and forth in the valley of Mountain Car">
  <figcaption><b>Figure 4.</b> Behavior of Q actor-critic (n = 1) in Mountain Car.</figcaption>
</figure>

Even after considerable parameter tuning and after running all of our agents in the Mountain Car environment, our implementation could not solve the problem. The agents never explored enough to reach the top even once. We assume that because policy gradient methods are on-policy and exploration is driven by the distribution over actions, the uniform reward of −1 is not enough to force sufficient exploration. Most of our agents behave as shown in the animation above.

### Comparing Random Seeds

<figure>
  <img src="{{ '/assets/rl_article/random_seed.png' | relative_url }}" alt="Two reward curves of the same agent with different random seeds">
  <figcaption><b>Figure 5.</b> One-step advantage actor-critic on CartPole with two different random seeds.</figcaption>
</figure>

To illustrate the variance across runs, we now look at two identical agents in the same environment whose only difference is the random seed. While the left agent solves the task almost perfectly and reaches the maximum possible reward (not visible due to smoothing), the right agent fails badly, with rewards between 5 and 10 throughout training. This shows how much policy gradient methods depend on a good initialization. It also shows another disadvantage of policy gradients: a bad policy leads to bad data. If the agent behaves suboptimally after some *wrong* updates in policy space, the data it collects will also be suboptimal, and in some cases the agent cannot recover.

### The Bias-Variance Debate

<figure>
  <img src="{{ '/assets/rl_article/variance_rewards_MC_A_Q.png' | relative_url }}" alt="Cumulative rewards with variance bands for three algorithms">
  <figcaption><b>Figure 6.</b> Cumulative rewards with variance over 5 runs during training for different algorithms.</figcaption>
</figure>

For the experiments in this section, we pick the most successful value of $$n$$ for advantage actor-critic and Q actor-critic and compare them with our baseline algorithm on CartPole-v0. To understand how our approaches differ with respect to the bias-variance trade-off, we look closely at the variance of the rewards and losses during training. The first figure shows the variance in rewards over the course of training. REINFORCE with baseline has the lowest variance overall; its variance is highest at the beginning and decreases during training. This means it learns the optimal behavior, which is likely because the environment is so simple. In contrast, for advantage actor-critic and especially for Q actor-critic, the variance in rewards is generally higher and even grows during training. Both algorithms are less stable than REINFORCE with baseline, and Q actor-critic is clearly the less stable of the two. Even though it occasionally "solves" the environment late in training, obtaining close to the maximum reward of 200, it also drops to rewards as low as 20. Advantage actor-critic, while more stable than Q actor-critic, is not as reliable as REINFORCE with baseline. Although it achieves fairly constant rewards between 150 and 200 late in training, it never seems to fully converge to the optimal policy. We interpret this as a sign of the bias of advantage actor-critic, since the unbiased REINFORCE with baseline does not show this behavior.

<figure>
  <img src="{{ '/assets/rl_article/variance_losses_MC_A_Q.png' | relative_url }}" alt="Actor and critic losses with variance bands for three algorithms">
  <figcaption><b>Figure 7.</b> Actor and critic losses with variance over 5 runs during training for different algorithms.</figcaption>
</figure>

Looking at the variance of the actor and critic losses, we first observe that all curves show rather high variance, which is not surprising. However, the REINFORCE actor loss starts low and grows the longer the episodes become. This makes sense, as the variance of the full sampled trajectories should be reflected in the loss. By contrast, the advantage actor-critic and Q actor-critic agents are much more homoscedastic in this respect. Another striking property of the actor loss is that it does not really decrease. This is reasonable if we recall the policy gradient update derived at the beginning: it is not a loss in the classical sense, but a procedure that increases the probability of well-performing actions.

The critic losses show rather normal loss behavior. We take the high loss of the Q-function as evidence for the weak performance of the Q actor-critic agents. This is underlined by the significantly lower loss of the advantage critic, which matches the superior performance of the advantage actor-critic agents. This suggests that the performance of an actor-critic method depends strongly on how well the critic converges.

### Larger N

We run one more experiment on CartPole to see how larger values of $$n$$ influence performance, with $$n \in \{1, 2, 4, 8, 32, 64\}$$.

<figure>
  <img src="{{ '/assets/rl_article/CP_n64.jpg' | relative_url }}" alt="Reward curves for larger n-step values on CartPole">
  <figcaption><b>Figure 8.</b> Performance of actor-critic methods with larger n on CartPole.</figcaption>
</figure>

The advantage actor-critic methods with $$n = 32$$ and $$n = 64$$ perform better than those with smaller $$n$$. This time, we found two values of $$n$$ that perform just as well as REINFORCE with baseline! For Q actor-critic, larger $$n$$ did not improve performance. One explanation for the positive effect of a large $$n$$ is that the algorithm behaves like REINFORCE as long as episodes are shorter than $$n$$ (early in training), while for longer episodes (late in training) it is a bit more stable than REINFORCE because it uses bootstrapping.

### Behavior

<figure>
  <img src="{{ '/assets/rl_article/cartpole.gif' | relative_url }}" alt="Agents balancing poles in CartPole">
  <figcaption><b>Figure 9.</b> Behavior of trained agents in CartPole.</figcaption>
</figure>

When we visualize the trained agents, we can observe some cool differences in behavior!

On CartPole, the one-step actor-critic agents typically fail to balance the pole for more than a few seconds and do not get a reward above 20. For $$n > 1$$, the advantage actor-critic agents do a decent job of balancing the pole, but they tend to drift to one side while balancing, which eventually makes them drop it. The REINFORCE with baseline agent learns to balance the pole well by making small movements from left to right.

On Lunar Lander, the agents take different approaches to landing the spaceship. The advantage actor-critic agent lands slowly by swaying from side to side (using the left and right engines) and lands better as $$n$$ increases, while the Q actor-critic agent comes crashing down in all experiments. The REINFORCE agent generally lands more directly, but it does not come to a halt after touching the surface and keeps firing its engines, which lowers its reward. See Figure 1 at the top of the page for this comparison!

Mountain Car is too challenging and does not lead to different behaviors. All agents stay stuck in the valley, rolling from left to right.

## Conclusion

All in all, our experiments show that increasing $$n$$ for advantage actor-critic methods can lead to performance almost as good as REINFORCE with baseline, but with less stability. Contrary to our hypothesis, tuning $$n$$ did not improve performance compared to the basic REINFORCE with baseline algorithm. As expected, the best value of $$n$$ depends on the environment.

Although our approach was successful in some cases, we generally observed a lot of variance in performance across algorithms, environments and even random seeds. This shows that policy gradients in deep reinforcement learning are complicated animals that demand considerable care and tuning. Our findings also support Ilyas et al.<sup class="cite"><a href="#ref-9">[9]</a></sup>, who state that in some prominent deep RL applications "performance depends heavily on optimizations not part of the core algorithm", which are not necessarily described in the papers. These optimizations include *reward scaling*, *learning rate annealing*, *orthogonal weight initialization* and *value clipping*, among others. We suspect that some of these optimizations are why the OpenAI Baselines implementation of A2C (even with a single environment) consistently solves Mountain Car, whereas our approach never did.

## Code

Our code is available [on GitHub](https://github.com/axlindt/variance_n_step_actor_critic).

## References

<ol class="refs">
  <li id="ref-1">Schulman et al. <a href="https://arxiv.org/abs/1506.02438">High-Dimensional Continuous Control Using Generalized Advantage Estimation</a>. ICLR 2016.</li>
  <li id="ref-2">OpenAI. <a href="https://openai.com/index/openai-five/">OpenAI Five</a>. 2018.</li>
  <li id="ref-3">OpenAI. <a href="https://openai.com/index/solving-rubiks-cube/">Solving Rubik's Cube with a Robot Hand</a>. 2019.</li>
  <li id="ref-4">Ilyas et al. <a href="https://gradientscience.org/policy_gradients_pt1/">A Closer Look at Deep Policy Gradients</a>. gradient science blog, 2018.</li>
  <li id="ref-5">Henderson et al. <a href="https://arxiv.org/abs/1709.06560">Deep Reinforcement Learning that Matters</a>. AAAI 2018.</li>
  <li id="ref-6">Sutton &amp; Barto. <a href="http://incompleteideas.net/book/RLbook2018.pdf">Reinforcement Learning: An Introduction</a>. 2nd edition, 2018.</li>
  <li id="ref-7">OpenAI. <a href="https://github.com/openai/baselines">OpenAI Baselines</a>.</li>
  <li id="ref-8">Mnih et al. <a href="https://arxiv.org/abs/1602.01783">Asynchronous Methods for Deep Reinforcement Learning</a>. ICML 2016.</li>
  <li id="ref-9">Ilyas et al. <a href="https://arxiv.org/abs/1811.02553">Are Deep Policy Gradient Algorithms Truly Policy Gradient Algorithms?</a> 2018.</li>
</ol>
