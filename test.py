import numpy as np
import torch
import gym
from PPO import PPO, device
from torch.utils.tensorboard import SummaryWriter
import os,shutil
from datetime import datetime
from rocket_gym import RocketMeister10
import pandas as pd

def Action_adapter(a,max_action):
    #from [0,1] to [-max,max]
    return  2*(a-0.5)*max_action

def evaluate_policy(env, model, render, steps_per_epoch, max_action):
    scores = 0
    turns = 3
    for j in range(turns):
        s, done, ep_r, steps = env.reset(), False, 0, 0
        while not (done or (steps >= steps_per_epoch)):
            # Take deterministic actions at test time
            a, logprob_a = model.evaluate(s)
            act = Action_adapter(a, max_action)  # [0,1] to [-max,max]
            s_prime, r, done, info = env.step(act)
            # r = Reward_adapter(r, EnvIdex)

            ep_r += r
            steps += 1
            s = s_prime
            if render:
                env.render()
        scores += ep_r
    return scores/turns

def main():


    write = False
    render = True
    Loadmodel= True
    env_with_Dead = True  #Env like 'LunarLanderContinuous-v2' is with Dead Signal. Important!
    T_horizon = 2048
    state_dim = 28
    action_dim = 2
    max_action = 1
    max_steps = 10000
    model_index=20200
    path = './model/3pi4'
    env_config = {
    'gui': True,
    'env_name': 'default',
    # 'env_name': 'empty',
    #'env_name': 'level1',
    #'env_name': 'level2',
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
    eval_env = RocketMeister10()


    Dist = ['Beta'] #type of probility distribution
    distnum = 0

    Max_episode = 100000
    save_interval = 200#in episode
    eval_interval = 10#in episode
    if write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format('rocket_meister') + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "env_with_Dead":env_with_Dead,
        "RNN":False,
        "gamma": 0.99,
        "lambd":0.95,     #For GAE
        "clip_rate": 0.2,  #0.2
        "K_epochs": 10,
        "net_width": 150,
        "a_lr": 2e-4,
        "c_lr": 2e-4,
        "dist": 'Beta',
        "l2_reg": 1e-3,   #L2 regulization for Critic
        "a_optim_batch_size":64,
        "c_optim_batch_size": 64,
        "entropy_coef":1e-3, #Entropy Loss for Actor: Large entropy_coef for large exploration, but is harm for convergence.
        "entropy_coef_decay":0.99
    }
    # if Dist[distnum] == 'Beta' :
    #     kwargs["a_lr"] *= 2 #Beta dist need large lr|maybe
    #     kwargs["c_lr"] *= 4  # Beta dist need large lr|maybe

    model = PPO(**kwargs)
    if Loadmodel: model.load(model_index,path)

    steps = 0
    total_steps = 0
    running_score = 0
    if Loadmodel:
        step_ = model_index
    else:
        step_ = 0
    for episode in range(100):
        s = env.reset()
        env.rocket.collision = False
        done = False
        ep_r = 0

        '''Interact & train'''
        for t in range(max_steps):
            steps+=1
            total_steps += 1

            if render:
                env.render()
                a, logprob_a = model.evaluate(s)
            else:
                a, logprob_a = model.select_action(s)

            act = Action_adapter(a,max_action) #[0,1] to [-max,max]
            s_prime, r, done, info = env.step(act)
            model.put_data((s, a, r, s_prime, logprob_a, done))
            s = s_prime
            ep_r += r
            '''update if its time'''
            if not render:
                if steps % T_horizon == 0:
                    print('update')
                    model.train()
                    steps = 0
            env.rocket.check_collision_env()
            if env.rocket.collision:
                break
            


        '''save model'''
        running_score = running_score * 0.99 + ep_r * 0.01


        '''record & log'''
        if (episode+1) % eval_interval == 0:
            score = evaluate_policy(eval_env, model, False, max_steps, max_action)
            print('EnvName:','Rocket_Meister','episode:', episode,'score:', score)

    env.close()

if __name__ == '__main__':
    main()

