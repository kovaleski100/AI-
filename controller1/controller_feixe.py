import controller_template as controller_template
import numpy as np
from operator import itemgetter
from random import *

class Controller(controller_template.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)
        self.last_checkpoit_distance = 0


    #######################################################################
    ##### METHODS YOU NEED TO IMPLEMENT ###################################
    #######################################################################

    def take_action(self, parameters: list) -> int:
        """
        :param parameters: Current weights/parameters of your controller
        :return: An integer corresponding to an action:
        1 - Right
        2 - Left
        3 - Accelerate
        4 - Brake
        5 - Nothing
        """

        features = self.compute_features(self.sensors)

        r = l = i = a = b = n = 0
        for feature in features:
            r += parameters[i]*feature
            i=i+1
        for feature in features:
            l += parameters[i]*feature
            i=i+1
        for feature in features:
            a += parameters[i]*feature
            i=i+1
        for feature in features:
            b += parameters[i]*feature
            i=i+1
        for feature in features:
            n += parameters[i]*feature
            i=i+1

        valor = 0
        valor = self.max_v(r, self.max_v(l, self.max_v(a, self.max_v(b,n))))

        if valor is r:
            #print("1")
            return 1
        if valor is l:
            #print("2")
            return 2
        if valor is a:
            #print("3")
            return 3
        if valor is b:
            #print("4")
            return 4
        if valor is n:
            #print("5")
            return 5

    def compute_features(self, sensors):
        """
        :param sensors: Car sensors at the current state s_t of the race/game
        contains (in order):
            track_distance_left: 0-100 (0 = tocando na grama, 100 = longe)
            track_distance_center: 0-100
            track_distance_right: 0-100
            on_track: 0 or 1
            checkpoint_distance: 0-???
            car_velocity: 10-200
            enemy_distance: -1 or 0-???
            position_angle: -180 to 180
            enemy_detected: 0 or 1
          (see the specification file/manual for more details)
        :return: A list containing the features you defined, in order
            left_feature
            right_feature
            checkpoint_feature
        """
        track_distance_left = max(1, sensors[0])
        track_distance_center = max(1, sensors[1])
        track_distance_right = max(1, sensors[2])
        on_track = sensors[3]
        max_on_track = 1
        min_on_track = 0
        checkpoint_distance = sensors[4]
        car_velocity = sensors[5]
        on_enemy_detected = sensors[6]
        #print(on_enemy_detected, "6")
        enemy_detected = sensors[7]
        #print(enemy_detected, "7")
        enemy_position = sensors[8]
        #print(enemy_position, "6")

        #if(on_enemy_detected):
        #    if(enemy_position<0):
        #        track_distance_left = 1
        #    elif(enemy_position>0):
        #        track_distance_right = 1

        #if(track_distance_left>track_distance_right):
        #    track_distance_right = 1
        #elif(track_distance_left<track_distance_right):
        #    track_distance_left = 1


        ## left feature
        left_feature = track_distance_left
        max_left_feature = 100
        min_left_feature = 1

        ## right feature
        right_feature = track_distance_right
        max_right_feature = 100
        min_right_feature = 1

        ## acclerate feature
        acclerate_feature = track_distance_center
        max_acclerate_feature = 100
        min_acclerate_feature = 1

        ## to the next check point feature
        ## se estamos nos aproximando retorna um numero negativo
        # checkpoint_feature = 0
        max_checkpoint_feature = 100 # TODO valores variáveis
        min_checkpoint_feature = -100 # TODO
        diff = checkpoint_distance - self.last_checkpoit_distance
        # if abs(diff) > max_checkpoint_feature: # aqui deu um pulo impossivel a não ser que tenha passsado por um checkpoint
            # checkpoint_feature = min_checkpoint_feature
        # else:
        checkpoint_feature = max(min_checkpoint_feature, min(max_checkpoint_feature, diff))
        self.last_checkpoit_distance = checkpoint_distance
        #####
        features = np.array([left_feature, acclerate_feature, right_feature, on_track,diff])
        max_values = np.array([max_left_feature, max_acclerate_feature, max_right_feature, max_on_track, max_checkpoint_feature])
        min_values = np.array([min_left_feature, min_acclerate_feature, min_right_feature, min_on_track, min_checkpoint_feature])

        ## normalize features
        for i in range(len(features)):
            features[i] = self.normalizeFeature(features[i], max_values[i], min_values[i])
        return features

    def viz(self, weights) -> list:
        v = []
        for weight in weights:
            ep = uniform(-0.1,0.1)
            new_w = weight + ep
            v.append(new_w)
        return v

    def learn(self, weights) -> list:

        v_iniciais = 4
        n = 20
        v = []
        melhores = []
        #melhores.append((weights,self.run_episode(weights)))
        for i in range(v_iniciais):
            rand = self.viz(weights)
            tup = (rand, self.run_episode(rand))
            melhores.append(tup)
        todos_v = []
        while True:
            for vi in melhores:
                for i in range(n):
                    N_rand = self.viz(vi[0])
                    rand_A = (N_rand,self.run_episode(N_rand))
                    todos_v.append(rand_A)
            todos_v = sorted(todos_v, key=itemgetter(1), reverse = True)
            melhores = sorted(melhores, key=itemgetter(1), reverse = True)

            if(todos_v[0][1]<melhores[0][1]):
                print("parou")
                print(melhores[0][1])
                return melhores[0][0]
            else:
                melhores = []
                for i in range(v_iniciais):
                    melhores.append(todos_v[i])
                print("new loop", " melhores", str(melhores[0][1]))
                todos_v = []

        """
        IMPLEMENT YOUR LEARNING METHOD (i.e. YOUR LOCAL SEARCH ALGORITHM) HERE

        HINT: you can call self.run_episode (see controller_template.py) to evaluate a given set of weights
        :param weights: initial weights of the controller (either loaded from a file or generated randomly)
        :return: the best weights found by your learning algorithm, after the learning process is over
        """
#        return melhores[0][0]
    def max_v(self, a, b):
        if(a>b):
            return a
        else:
            return b
    def min_v(self, a, b):
        if(a>b):
            return b
        else:
            return a
        #raise NotImplementedError("This Method Must Be Implemented")
    def normalizeFeature(self, value, max, min):
        return 2 * (value - min)/(max - min) - 1
