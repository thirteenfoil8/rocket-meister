import numpy as np
import torch
import gym
from SAC import SAC_Agent
from ReplayBuffer import RandomBuffer, device
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os, shutil
import argparse
from rocket_gym import RocketMeister10

def Action_adapter(a,max_action,model='PPO'):
    return  a*max_action

def Action_adapter_reverse(act,max_action):
    #from [-max,max] to [-1,1]
    return  act/max_action

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def evaluate_policy(env, model, render, steps_per_epoch, max_action):
    scores = 0
    turns = 3
    for j in range(turns):
        s, done, ep_r = env.reset(), False, 0
        while not done:
            # Take deterministic actions at test time
            a = model.select_action(s, deterministic=True, with_logprob=False)
            act = Action_adapter(a, max_action)  # [0,1] to [-max,max]
            s_prime, r, done, info = env.step(act)
            # r = Reward_adapter(r, EnvIdex)
            ep_r += r
            s = s_prime
            if render:
                env.render()
        # print(ep_r)
        scores += ep_r
    return scores/turns

def main():

    write = True   #Use SummaryWriter to record the training.
    render = False
    Loadmodel= False

    # Env config:
    EnvName = 'Rocket_Meister'
    env_with_Dead = True
    model_index=16000
    max_steps= 1000
    env_config = {
    'gui': True,
    'env_name': 'default',
    # 'env_name': 'empty',
    # 'env_name': 'level1',
    # 'env_name': 'level2',
    #'env_name': 'random',
    #'camera_mode': 'centered',
    # 'env_flipped': False,
    # 'env_flipmode': False,
    # 'export_frames': True,
    'export_states': False,
    # 'export_highscore': False,
    'export_string': 'ppo',
    'max_steps': max_steps,
    'gui_reward_total': True,
    'gui_echo_distances': True,
    'gui_level': True,
    'gui_velocity': True,
    'gui_goal_ang': True,
    'gui_frames_remaining': True,
    'gui_draw_echo_points': True,
    'gui_draw_echo_vectors': True,
    'gui_draw_goal_points': True,
    }
    env = RocketMeister10(env_config)
    eval_env = RocketMeister10(env_config)
    state_dim = 28
    action_dim = 2
    max_action = 1
    steps_per_epoch = 100
    print('Env:','Rocket_Meister','  state_dim:',state_dim,'  action_dim:',action_dim,
          '  max_a:',max_action,'  min_a:',-max_action, 'max_episode_steps', steps_per_epoch)

    #Interaction config:
    start_steps = 5*steps_per_epoch #in steps
    update_after = 2*steps_per_epoch #in steps
    update_every = 50
    total_steps = 1000000
    eval_interval = 10  #in steps
    save_interval = 200  #in steps
    running_score = 0


    #SummaryWriter config:
    if write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/SAC_{}'.format('Rocket_Meister') + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)



    #Model hyperparameter config:
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": 0.99,
        "hid_shape": (256,256),
        "a_lr": 1e-4,
        "c_lr": 1e-4,
        "batch_size":256,
        "alpha":1,
        "adaptive_alpha":True,
        "tau":0.005 # put 0.005 if easy to train, else 1
    }


    model = SAC_Agent(**kwargs)
    if not os.path.exists('model'): os.mkdir('model')
    if Loadmodel: model.load(model_index)
    if Loadmodel:
        step_ = model_index
    else:
        step_ = 0

    replay_buffer = RandomBuffer(state_dim, action_dim, env_with_Dead, max_size=int(1e6))

    if render:
        average_reward = evaluate_policy(env, model, render, steps_per_epoch, max_action)
        print('Average Reward:', average_reward)
    else:
        s, done, current_steps = env.reset(), False, 0
        for t in range(step_,total_steps):
            current_steps += 1
            env.rocket.collision = False
            '''Interact & trian'''

            if t < start_steps:
                #Random explore for start_steps
                act = env.action_space.sample() #act∈[-max,max]
                a = Action_adapter_reverse(act,max_action) #a∈[-1,1]
            else:
                a = model.select_action(s, deterministic=False, with_logprob=False) #a∈[-1,1]
                act = Action_adapter(a,max_action) #act∈[-max,max]

            s_prime, r, done, info = env.step(act)
            done= env.rocket.done
            replay_buffer.add(s, a, r, s_prime, done)
            s = s_prime


            # 50 environment steps company with 50 gradient steps.
            # Stabler than 1 environment step company with 1 gradient step.
            if t >= update_after and t % update_every == 0 :
                for j in range(update_every):
                    model.train(replay_buffer)
                print('update')
            
            '''save model'''
            if (t + 1) % save_interval == 0:
                model.save(t + 1)
                try:
                    replay_buffer.save()
                except:
                    continue

            '''record & log'''
            if (t + 1) % eval_interval == 0:
                score = evaluate_policy(eval_env, model, False, steps_per_epoch, max_action)
                running_score = running_score * 0.99 + score * 0.01
                if write:
                    writer.add_scalar('running_score', running_score, global_step=t + 1)
                    writer.add_scalar('ep_r', score, global_step=t + 1)
                    writer.add_scalar('alpha', model.alpha, global_step=t + 1)
                print('EnvName:', 'Rocket_Meister', 'totalsteps:', t+1, 'score:', score)
            
            if done:
                s, done, current_steps = env.reset(), False, 0

    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()
