# Effective-Drone-Navigation

Create a python environment in your local desktop to run the file 
A  reinforcement Learning model for effective path and trajectory planning to find suitable camera views for optimal object detection.
This is a project about deep reinforcement learning autonomous obstacle avoidance algorithm for UAV. The whole project includes obstacle avoidance in static environment and obstacle avoidance in dynamic environment. In the static environment, Multi-Agent Reinforcement Learning and artificial potential field algorithm are combined. In the dynamic environment, the project adopts the combination of disturbed flow field algorithm and single agent reinforcement learning algorithm.



## Requirements:

numpy

torch

matplotlib

seaborn==0.11.1



## How to begin trainning:

For example, you want to train the agent in dynamic environment with TD3, what you need to do is just running the 
main.py, && test.py,

finally open matlab and run the test.m to draw.

If you want to test the model in the environment with 4 obstacles, you just need to run : Multi_obstacle_environment_test.py.




## Files to illustrate:

calGs.m: calculate the index Gs which shows the performance of the route.

calLs.m: calculate the index Ls which shows the performance of the route.

draw.py: this file includes the Painter class which can draw the reward curve of various methods.

config.py: this file give the setting of the parameters in trainning process of the algorithm such as the MAX_EPISODE, batch_size and so on.

Method.py: this file concludes many important methods such as how to calculate the reward of the agents.

static_obstacle_environment.py: there are many static obstacle environments' parameters in this file.

dynamic_obstacle_environment.py: there are many dynamic obstacle environments' parameters in this file.

Multi_obstacle_environment_test.py: this file test the dynamic model in the environment in dynamic_obstacle_environment.py.

data_csv: this file save some data such as the trace of UAV and the reward in trainning.


# ====AI ARCHITECTS====
1. Tapan Mahata
2. Dev Wankhede
3. Akshat Dubey
4. Abhinit Singh
5. Pradoom Varma 

