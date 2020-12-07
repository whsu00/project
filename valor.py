 # VALOR implementation 
import sys
sys.path.insert(0, './utils/')
import numpy as np 
import torch
import torch.nn.functional as F 
import gym 
import time
import scipy.signal
from network import Discriminator, ActorCritic, count_vars
from buffer import Buffer
from torch.distributions.categorical import Categorical
# from utils.mpi_tools import mpi_fork, proc_id, mpi_statistics_scalar, num_procs
# from utils.mpi_torch import average_gradients, sync_all_params
from logx import EpochLogger
import pytorch_utils as ptu
from types import SimpleNamespace
import pdb

def valor(args):

    '''
    Notes: Discriminator is the decoder
    TODO: 1) Go through policy update
        2) Write out flow to yourself
        3) Implement Curriculum Learning
        4) Scrutinize possible mistakes (why is logp instead of log_gt stored in a lot of places)
            a) Why isn't LSTM Policy used in encoder?
            b) Why is policy loss the way it is?
            c) Why is expectation taken the way that it is in decoder?

    '''
    '''
    Major Bug:
    Buffer gets overriden every 5 timesteps -> because indeces get reset everytime you pull from the buffer
    '''
    env_fn = args.get('env_fn', lambda: gym.make('HalfCheetah-v2'))
    actor_critic = args.get('actor_critic', ActorCritic)
    ac_kwargs = args.get('ac_kwargs', {})
    disc = args.get('disc', Discriminator)
    dc_kwargs = args.get('dc_kwargs', {})
    seed = args.get('seed', 0)
    episodes_per_epoch = args.get('episodes_per_epoch', 40)
    epochs = args.get('epochs', 50)
    gamma = args.get('gamma', 0.99)
    pi_lr = args.get('pi_lr', 3e-4)
    vf_lr = args.get('vf_lr', 1e-3)
    dc_lr = args.get('dc_lr', 2e-3)
    train_v_iters = args.get('train_v_iters', 80)
    train_dc_iters = args.get('train_dc_iters', 50)
    train_dc_interv = args.get('train_dc_interv', 2)
    lam = args.get('lam', 0.97)
    max_ep_len = args.get('max_ep_len', 1000)
    logger_kwargs = args.get('logger_kwargs', {})
    context_dim = args.get('context_dim', 4)
    max_context_dim = args.get('max_context_dim', 64)
    save_freq = args.get('save_freq', 10)
    k = args.get('k', 1)

    # GPU handling
    ptu.init_gpu(args.get('use_gpu', False), args.get('gpu_id', 0))

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac_kwargs['action_space'] = env.action_space

    # Model
    actor_critic = actor_critic(input_dim=obs_dim[0]+max_context_dim, **ac_kwargs)
    disc = disc(input_dim=obs_dim[0], context_dim=max_context_dim, **dc_kwargs)

    # Buffer
    local_episodes_per_epoch = episodes_per_epoch # int(episodes_per_epoch / num_procs())
    buffer = Buffer(max_context_dim, obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len, train_dc_interv, gamma=gamma, lam=lam)

    # Count variables
    var_counts = tuple(count_vars(module) for module in
        [actor_critic.policy, actor_critic.value_f, disc.policy])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t d: %d\n'%var_counts)    

    # Optimizers
    #Optimizer for RL Policy
    train_pi = torch.optim.Adam(actor_critic.policy.parameters(), lr=pi_lr)

    #Optimizer for value function (for actor-critic)
    train_v = torch.optim.Adam(actor_critic.value_f.parameters(), lr=vf_lr)

    #Optimizer for decoder
    train_dc = torch.optim.Adam(disc.policy.parameters(), lr=dc_lr)

    # Parameters Sync
    # sync_all_params(actor_critic.parameters())
    # sync_all_params(disc.parameters())

    '''
    Training function
    '''
    def update(e):
        obs, act, adv, pos, ret, logp_old = [ptu.from_numpy(x) for x in buffer.retrieve_all()]
        
        # Policy
        _, logp, _ = actor_critic.policy(obs, act)
        entropy = (-logp).mean()

        # Policy loss
        pi_loss = -(logp*(k*adv+pos)).mean()

        # Train policy (Go through policy update)
        train_pi.zero_grad()
        pi_loss.backward()
        # average_gradients(train_pi.param_groups)
        train_pi.step()

        # Value function
        v = actor_critic.value_f(obs)
        v_l_old = F.mse_loss(v, ret)
        for _ in range(train_v_iters):
            v = actor_critic.value_f(obs)
            v_loss = F.mse_loss(v, ret)

            # Value function train
            train_v.zero_grad()
            v_loss.backward()
            # average_gradients(train_v.param_groups)
            train_v.step()

        # Discriminator
        if (e+1) % train_dc_interv == 0:
            print('Discriminator Update!')
            con, s_diff = [ptu.from_numpy(x) for x in buffer.retrieve_dc_buff()]
            _, logp_dc, _ = disc(s_diff, con)
            d_l_old = -logp_dc.mean()

            # Discriminator train
            for _ in range(train_dc_iters):
                _, logp_dc, _ = disc(s_diff, con)
                d_loss = -logp_dc.mean()
                train_dc.zero_grad()
                d_loss.backward()
                # average_gradients(train_dc.param_groups)
                train_dc.step()

            _, logp_dc, _ = disc(s_diff, con)
            dc_l_new = -logp_dc.mean()
        else:
            d_l_old = 0
            dc_l_new = 0

        # Log the changes
        _, logp, _, v = actor_critic(obs, act)
        pi_l_new = -(logp*(k*adv+pos)).mean()
        v_l_new = F.mse_loss(v, ret)
        kl = (logp_old - logp).mean()
        logger.store(LossPi=pi_loss, LossV=v_l_old, KL=kl, Entropy=entropy, DeltaLossPi=(pi_l_new-pi_loss),
            DeltaLossV=(v_l_new-v_l_old), LossDC=d_l_old, DeltaLossDC=(dc_l_new-d_l_old))
        # logger.store(Adv=adv.reshape(-1).numpy().tolist(), Pos=pos.reshape(-1).numpy().tolist())

    start_time = time.time()
    #Resets observations, rewards, done boolean
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    #Creates context distribution where each logit is equal to one (This is first place to make change)
    init_context_prob_arr = np.array(context_dim* [1/context_dim] + (max_context_dim - context_dim)*[0])
    context_dist = Categorical(probs=ptu.from_numpy(init_context_prob_arr))
    total_t = 0

    for epoch in range(epochs):
        #Sets actor critic and decoder (discriminator) into eval mode
        actor_critic.eval()
        disc.eval()
        
        #Runs the policy local_episodes_per_epoch before updating the policy
        for index in range(local_episodes_per_epoch):
            # Sample from context distribution and one-hot encode it (Step 2)
            # Every time we run the policy we sample a new context

            if epoch == 52:
                pdb.set_trace()
    
            c = context_dist.sample()

            c_onehot = F.one_hot(c, max_context_dim).squeeze().float()
            for _ in range(max_ep_len):
                concat_obs = torch.cat([ptu.from_numpy(o.reshape(1, -1)), c_onehot.reshape(1, -1)], 1)
                '''
                Feeds in observation and context into actor_critic which spits out a distribution 
                Label is a sample from the observation
                pi is the action sampled
                logp is the log probability of some other action a
                logp_pi is the log probability of pi 
                v_t is the value function
                '''
                a, _, logp_t, v_t = actor_critic(concat_obs)

                #Stores context and all other info about the state in the buffer
                buffer.store(c, ptu.to_numpy(concat_obs.squeeze()), ptu.to_numpy(a), r, v_t.item(), ptu.to_numpy(logp_t))
                logger.store(VVals=v_t)

                o, r, d, _ = env.step(ptu.to_numpy(a)[0])
                ep_ret += r
                ep_len += 1
                total_t += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal:
                    # Key stuff with discriminator
                    dc_diff = ptu.from_numpy(buffer.calc_diff()).unsqueeze(0)
                    #Context
                    con = ptu.from_numpy(np.array([float(c)])).unsqueeze(0)
                    #Feed in differences between each state in your trajectory and a specific context
                    #Here, this is just the log probability of the label it thinks it is
                    _, _, log_p = disc(dc_diff, con)
                    buffer.end_episode(ptu.to_numpy(log_p))
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [actor_critic, disc], None)

        # Sets actor_critic and discriminator into training mode
        actor_critic.train()
        disc.train()

        update(epoch)
        #Need to implement curriculum learning here to update context distribution 
        ''' 
            #Psuedocode:
            Loop through each of d episodes taken in local_episodes_per_epoch and check log probability from discrimantor
            If >= 0.86, increase k in the following manner: k = min(int(1.5*k + 1), Kmax)
            Kmax = 64
        '''
        if (epoch + 1 )% train_dc_interv == 0 and epoch > 0:
            con, s_diff = [ptu.from_numpy(x) for x in buffer.retrieve_dc_buff()]
            print("Context: ",  con)
            # print("State Diffs", s_diff)
            print("num_contexts", len(con))
            _, logp_dc, _ = disc(s_diff, con)
            log_p_context_sample = ptu.to_numpy(logp_dc.mean())

            print("Log Probability context sample", log_p_context_sample)

            decoder_accuracy = np.exp(log_p_context_sample)
            print("Decoder Accuracy", decoder_accuracy)

            if decoder_accuracy >= 0.5:
                new_context_dim = min(int(1.5*context_dim+1), max_context_dim)
                print("new_context_dim: ", new_context_dim)
                new_context_prob_arr = np.array(new_context_dim * [1/new_context_dim] + (max_context_dim - new_context_dim)*[0])
                context_dist = Categorical(probs=ptu.from_numpy(new_context_prob_arr))
                context_dim = new_context_dim

            buffer.clear_dc_buff()


        # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', total_t)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('LossDC', average_only=True)
        logger.log_tabular('DeltaLossDC', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='valor')
    parser.add_argument('--con', type=int, default=5)
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    # mpi_fork(args.cpu)

    from run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)


    valor(args)
