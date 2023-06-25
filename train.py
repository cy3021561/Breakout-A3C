# Training the AI

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, params, shared_model, optimizer):
    # Shift the seeds that each training agent is desynchronized
    # rank is the agent amount
    torch.manual_seed(params.seed + rank)
    env = create_atari_env(params.env_name)
    env.seed(params.seed + rank)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    episode_length = 0
    while True:
        episode_length += 1
        # Synchronize the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(params.num_steps):
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(action_values)
            log_prob = F.log_softmax(action_values)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial().data
            # Get log prob based on the action
            log_prob = log_prob.gather(1, Variable(action))
            values.append(value)
            log_probs.append(log_prob)
            
            # Play the action
            state, reward, done, _ = env.step(action.numpy())
            done = (done or episode_length >= params.max_episode_length)
            # Clamp the reward between -1 and 1
            reward = max(min(reward, 1), -1)
            if done:
                episode_length = 0
                state = env.reset()
            state = torch.from_numpy(state)
            rewards.append(reward)
            if done:
                break
        # Initialize the cumulative reward
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data
        values.append(Variable(R))
        # Initialize the losses
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        # Generalized advantage estimation -> A(a, s) = Q(a, s) - V(s)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            # R = gamma*R + r_t = r_0 + gamma r_1 + gamma^2 * r_2 ... + gamma^(n-1)*r_(n-1) + gamma^nb_step * V(last_state)
            R = params.gamma * R + rewards[i]
            # R is an estimator of Q at time t = i so advantage_i = Q_i - V(state_i) = R - value[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2) # Q*(a*, s) should equal to V*(s), * means optimal
            # Temporal difference
            TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data
            # gae = sum_i (gamma*tau)^i * TD(i) with gae_i = gae_(i+1)*gamma*tau + (r_i + gamma*V(state_i+1) - V(state_i))
            gae = gae * params.gamma * params.tau + TD
            # policy_loss = - sum_i log(policy_i) * gae + 0.01 * entropy_i
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        # clamping the values of gradient between 0 and 40 to prevent the gradient from taking huge values and degenerating the algorithm
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)
        optimizer.step()
