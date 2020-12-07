 # VALOR implementation 
import numpy as np 
import torch
import torch.nn.functional as F 
import gym 
import time
import scipy.signal
from network import Discriminator, ActorCritic, count_vars
from buffer import Buffer
from torch.distributions.categorical import Categorical
#from utils.mpi_tools import mpi_fork, proc_id, mpi_statistics_scalar, num_procs
#from utils.mpi_torch import average_gradients, sync_all_params
from utils.logx import EpochLogger
import pdb

def valor(env_fn, actor_critic=ActorCritic, ac_kwargs=dict(), disc=Discriminator, dc_kwargs=dict(), seed=0, episodes_per_epoch=40,
        epochs=50, gamma=0.99, pi_lr=3e-4, vf_lr=1e-3, dc_lr=2e-3, train_v_iters=80, train_dc_iters=50, train_dc_interv=2, 
        lam=0.97, max_ep_len=40, logger_kwargs=dict(), con_dim=4, max_con_dim = 64, save_freq=10, k=1):

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
    actor_critic = actor_critic(input_dim=obs_dim[0]+max_con_dim, **ac_kwargs)
    disc = disc(input_dim=obs_dim[0], context_dim=max_con_dim, **dc_kwargs)

    # Buffer
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    buffer = Buffer(max_con_dim, obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len, train_dc_interv)

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

    #pdb.set_trace()

    # Parameters Sync
    #sync_all_params(actor_critic.parameters())
    #sync_all_params(disc.parameters())

    '''
    Training function
    '''
    def update(e):
        obs, act, adv, pos, ret, logp_old = [torch.Tensor(x) for x in buffer.retrieve_all()]
        
        # Policy
        #pdb.set_trace()
        _, logp, _ = actor_critic.policy(obs, act, batch = False)
        #pdb.set_trace()
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
            con, s_diff = [torch.Tensor(x) for x in buffer.retrieve_dc_buff()]
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
    context_dim_prob_dict = {i: 1/con_dim if i < con_dim else 0 for i in range(max_con_dim)} 
    last_phi_dict = {i:0 for i in range(con_dim)}
    context_dist = Categorical(probs=torch.Tensor(list(context_dim_prob_dict.values())))
    total_t = 0

    for epoch in range(epochs):
        #Sets actor critic and decoder (discriminator) into eval mode
        actor_critic.eval()
        disc.eval()
        
        #Runs the policy local_episodes_per_epoch before updating the policy
        for index in range(local_episodes_per_epoch):
            # Sample from context distribution and one-hot encode it (Step 2)
            # Every time we run the policy we sample a new context
    
            c = context_dist.sample()
            c_onehot = F.one_hot(c, max_con_dim).squeeze().float()
            for _ in range(max_ep_len):
                concat_obs = torch.cat([torch.Tensor(o.reshape(1, -1)), c_onehot.reshape(1, -1)], 1)
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
                buffer.store(c, concat_obs.squeeze().detach().numpy(), a.detach().numpy(), r, v_t.item(), logp_t.detach().numpy())
                logger.store(VVals=v_t)

                o, r, d, _ = env.step(a.detach().numpy()[0])
                ep_ret += r
                ep_len += 1
                total_t += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal:
                    # Key stuff with discriminator
                    dc_diff = torch.Tensor(buffer.calc_diff()).unsqueeze(0)
                    #Context
                    con = torch.Tensor([float(c)]).unsqueeze(0)
                    #Feed in differences between each state in your trajectory and a specific context
                    #Here, this is just the log probability of the label it thinks it is
                    _, _, log_p = disc(dc_diff, con)
                    buffer.end_episode(log_p.detach().numpy())
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
            #pdb.set_trace()
            con, s_diff = [torch.Tensor(x) for x in buffer.retrieve_dc_buff()]
            print("Context: ",  con)
            print("num_contexts", len(con))
            _, logp_dc, _ = disc(s_diff, con)
            log_p_context_sample = logp_dc.mean().detach().numpy()

            print("Log Probability context sample", log_p_context_sample)

            decoder_accuracy = np.exp(log_p_context_sample)
            print("Decoder Accuracy", decoder_accuracy)

            logger.store(LogProbabilityContext = log_p_context_sample, DecoderAccuracy = decoder_accuracy)

            '''
            Create score (phi(i)) = -log_p_context_sample.mean() for each specific context 
            Assign phis to each specific context
            Get p(i) in the following manner: (phi(i) + epsilon)
            Get Probabilities by doing p(i)/sum of all p(i)'s 
            '''
            logp_np = logp_dc.detach().numpy()
            con_np = con.detach().numpy()
            phi_dict = {i:0 for i in range(con_dim)}
            count_dict = {i:0 for i in range(con_dim)}
            for i in range(len(logp_np)):
                current_con = con_np[i]
                phi_dict[current_con] += logp_np[i]
                count_dict[current_con] += 1
            print(phi_dict)
            
            phi_dict = {k: last_phi_dict[k] if count_dict[k] == 0 else (-1) * v/count_dict[k] for (k,v) in phi_dict.items()}
            sorted_dict = dict(sorted(phi_dict.items(), key=lambda item: item[1], reverse = True))
            sorted_dict_keys = list(sorted_dict.keys())
            rank_dict = {sorted_dict_keys[i]: 1/(i+1) for i in range(len(sorted_dict_keys))}
            rank_dict_sum = sum(list(rank_dict.values()))
            context_dim_prob_dict = {k:rank_dict[k]/rank_dict_sum if k < con_dim else 0 for k in context_dim_prob_dict.keys()}
            print(context_dim_prob_dict)

            if decoder_accuracy >= 0.85:
                new_context_dim = min(int(1.5*con_dim+1), max_con_dim)
                # print("new_context_dim: ", new_context_dim)
                new_context_prob_arr = new_context_dim * [1/new_context_dim] + (max_con_dim - new_context_dim)*[0]
                context_dist = Categorical(probs=torch.Tensor(new_context_prob_arr))
                con_dim = new_context_dim

            for i in range(con_dim):
              if i in phi_dict:
                last_phi_dict[i] = phi_dict[i]
              elif i not in last_phi_dict:
                last_phi_dict[i] = max(phi_dict.values())
            
            buffer.clear_dc_buff()
        else:
          logger.store(LogProbabilityContext = 0, DecoderAccuracy = 0)

        # Log
        logger.store(ContextDim = con_dim)
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
        logger.log_tabular('LogProbabilityContext', average_only=True)
        logger.log_tabular('DecoderAccuracy', average_only = True)
        logger.log_tabular('ContextDim', average_only = True)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='valor')
    parser.add_argument('--con', type=int, default=4)
    args = parser.parse_args()

    #mpi_fork(args.cpu)

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    valor(lambda: gym.make(args.env), actor_critic=ActorCritic, ac_kwargs=dict(hidden_dims=[args.hid]*args.l, lstmFlag = False),
        disc=Discriminator, dc_kwargs=dict(hidden_dims=args.hid),  gamma=args.gamma, seed=args.seed, episodes_per_epoch=args.episodes, epochs=args.epochs, logger_kwargs=logger_kwargs, con_dim=args.con)

