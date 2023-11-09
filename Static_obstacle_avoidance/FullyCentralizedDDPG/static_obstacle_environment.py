

import numpy as np
class Obstacle1:
    def __init__(self):
        self.obstacle = np.array([[3, 2, 1],
                                  [3, 4, 2],
                                  [5, 4, 2],
                                  [6, 2, 1]
                                  ], dtype=float)  
        self.Robstacle = np.array([1, 2, 2, 1], dtype=float)  
        self.cylinder = np.array([[8, 3]
                                  ], dtype=float)  
        self.cylinderR = np.array([0.5], dtype=float)  
        self.cylinderH = np.array([8], dtype=float)  
        self.cone = np.array([[4, 0]
                              ], dtype=float)  
        self.coneR = np.array([1], dtype=float)  
        self.coneH = np.array([3], dtype=float)  
        self.qgoal = np.array([8, 6, 1.2], dtype=float)  
        self.x0 = np.array([0, 0, 2], dtype=float)  

class Obstacle2:
    def __init__(self):
        self.obstacle = np.array([[2, 3, 2],
                                  [4, 7, 1],
                                  [7, 7, 2]
                                  ], dtype=float)  
        self.Robstacle = np.array([2, 1, 2], dtype=float)  
        self.cylinder = np.array([[5, 5],
                                  [8, 4]
                                  ], dtype=float)  
        self.cylinderR = np.array([1, 1], dtype=float)  
        self.cylinderH = np.array([7, 6], dtype=float)  
        self.cone = np.array([[5, 2]
                              ], dtype=float)  
        self.coneR = np.array([2], dtype=float)  
        self.coneH = np.array([3], dtype=float)  

        self.qgoal = np.array([10, 7, 1], dtype=float)  
        self.x0 = np.array([-2, 1, 3], dtype=float)  

class Obstacle3:
    def __init__(self):
        self.obstacle = np.array([[2, 2, 1],
                                  [2, 4, 1],
                                  [4, 2, 1],
                                  [4, 4, 1],
                                  [6, 2, 1],
                                  [6, 4, 1]
                                  ], dtype=float)  
        self.Robstacle = np.array([1, 1, 1, 1, 1, 1], dtype=float)  
        self.cylinder = np.array([
        ], dtype=float)  
        self.cylinderR = np.array([], dtype=float)  
        self.cylinderH = np.array([], dtype=float)  
        self.cone = np.array([
        ], dtype=float)  
        self.coneR = np.array([], dtype=float)  
        self.coneH = np.array([], dtype=float)  

        self.qgoal = np.array([8, 6, 1.5], dtype=float)  
        self.x0 = np.array([0, 0, 1], dtype=float)  

class Obstacle4:
    def __init__(self):
        self.obstacle = np.array([[2, 5, 2],
                                  [4, 3, 2],
                                  [8, 8, 2]
                                  ], dtype=float)  
        self.Robstacle = np.array([2, 2, 2], dtype=float)  
        self.cylinder = np.array([[4, 7]
                                  ], dtype=float)  
        self.cylinderR = np.array([1], dtype=float)  
        self.cylinderH = np.array([6], dtype=float)  
        self.cone = np.array([[8, 5]
                              ], dtype=float)  
        self.coneR = np.array([2], dtype=float)  
        self.coneH = np.array([4], dtype=float)  

        self.qgoal = np.array([10, 11, 2], dtype=float)  
        self.x0 = np.array([0, 0, 3], dtype=float)  

Obstacle = {"Obstacle1":Obstacle1(),"Obstacle2":Obstacle2(),"Obstacle3":Obstacle3(),"Obstacle4":Obstacle4()}