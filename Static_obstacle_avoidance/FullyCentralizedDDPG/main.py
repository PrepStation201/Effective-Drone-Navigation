from DDPGModel import DDPG
from ApfAlgorithm import APF
from Method import getReward, setup_seed
import random
import numpy as np
from draw import Painter
import torch as torch

if __name__ == '__main__':
    setup_seed(11)   

    apf = APF()
    obs_dim = 6 * (apf.numberOfSphere + apf.numberOfCylinder + apf.numberOfCone)
    act_dim = 1 * (apf.numberOfSphere + apf.numberOfCylinder + apf.numberOfCone)
    act_bound = [0.1, 3]

    centralizedContriller = DDPG(obs_dim, act_dim, act_bound)

    MAX_EPISODE = 1000
    MAX_STEP = 500
    update_every = 50
    batch_size = 128
    noise = 0.3
    update_cnt = 0
    rewardList = []
    maxReward = -np.inf
    for episode in range(MAX_EPISODE):
        q = apf.x0
        apf.reset()
        rewardSum = 0
        qBefore = [None, None, None]
        for j in range(MAX_STEP):
            obsDicq = apf.calculateDynamicState(q)
            obs_sphere, obs_cylinder, obs_cone = obsDicq['sphere'], obsDicq['cylinder'], obsDicq['cone']
            obs_mix = obs_sphere + obs_cylinder + obs_cone
            obs = np.array([]) 
            for k in range(len(obs_mix)):
                obs = np.hstack((obs, obs_mix[k])) 
            if episode > 50:
                noise *= 0.99995
                if noise <= 0.1: noise = 0.1
                action = centralizedContriller.get_action(obs, noise_scale=noise)
                
                action_sphere = action[0:apf.numberOfSphere]
                action_cylinder = action[apf.numberOfSphere:apf.numberOfSphere + apf.numberOfCylinder]
                action_cone = action[apf.numberOfSphere + apf.numberOfCylinder:apf.numberOfSphere +\
                                          apf.numberOfCylinder + apf.numberOfCone]
            else:
                action_sphere = [random.uniform(act_bound[0],act_bound[1]) for k in range(apf.numberOfSphere)]
                action_cylinder = [random.uniform(act_bound[0],act_bound[1]) for k in range(apf.numberOfCylinder)]
                action_cone = [random.uniform(act_bound[0],act_bound[1]) for k in range(apf.numberOfCone)]
                action = action_sphere + action_cylinder + action_cone

            
            qNext = apf.getqNext(apf.epsilon0, action_sphere, action_cylinder, action_cone, q, qBefore)
            obsDicqNext = apf.calculateDynamicState(qNext)
            obs_sphere_next, obs_cylinder_next, obs_cone_next = obsDicqNext['sphere'], obsDicqNext['cylinder'], obsDicqNext['cone']
            obs_mix_next = obs_sphere_next + obs_cylinder_next + obs_cone_next
            obs_next = np.array([])
            for k in range(len(obs_mix_next)):
                obs_next = np.hstack((obs_next, obs_mix_next[k]))
            flag = apf.checkCollision(qNext)
            reward = getReward(flag, apf, qBefore, q, qNext)
            rewardSum += reward

            done = True if apf.distanceCost(apf.qgoal, qNext) < apf.threshold else False
            centralizedContriller.replay_buffer.store(obs, action, reward, obs_next, done)

            if episode >= 50 and j % update_every == 0:
                if centralizedContriller.replay_buffer.size >= batch_size:
                    update_cnt += update_every
                    for _ in range(update_every):
                        batch = centralizedContriller.replay_buffer.sample_batch(batch_size)
                        centralizedContriller.update(data=batch)
            if done: break
            qBefore = q
            q = qNext


        print('Episode:', episode, 'Reward:%f' % rewardSum, 'noise:%f' % noise, 'update_cnt:%d' %update_cnt)
        rewardList.append(round(rewardSum,2))
        if episode > MAX_EPISODE*2/3:
            if rewardSum > maxReward:
                print('Reward greater than the historical best, model saved!')
                maxReward = rewardSum
                torch.save(centralizedContriller.ac.pi, 'TrainedModel/centralizedActor.pkl')

    
    painter = Painter(load_csv=True,load_dir='/home/tapancs/Desktop/Effective-Drone-Navigation/UAV_Obstacle_Avoiding_DRL/Static_obstacle_avoidance/FullyCentralizedDDPG/data_csv/figure_data.csv')
    painter.addData(rewardList, 'Fully Centralized DDPG',smooth=True)
    painter.saveData('/home/tapancs/Desktop/Effective-Drone-Navigation/UAV_Obstacle_Avoiding_DRL/Static_obstacle_avoidance/FullyCentralizedDDPG/data_csv/figure_data.csv')
    painter.drawFigure()





