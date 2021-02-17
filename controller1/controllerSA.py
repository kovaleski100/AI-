import controller_template as controller_template
import matplotlib.pyplot as plt
import numpy as np
import math, random

class Controller(controller_template.Controller):

    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)
        self.last_checkpoit_distance = -1000;

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

        # According to the formula, there is a theta, for each action,
        # that is independent of feature value
        # Insert 1 at beginning to add the independent theta value
        features = [1] + self.compute_features(self.sensors)

        Q = [0, 0, 0, 0, 0]
        for q in range(len(Q)):
            for f in range(len(features)):
                # print("Tipo: ", type(features[f]))
                # print("Tipo: ", type())
                Q[q] += features[f] * parameters[q*len(features)+f]
        return 1 + Q.index(max(Q)) # Because starts with 1


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
        checkpoint_distance = sensors[4]
        car_velocity = sensors[5]
        
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
        features = np.array([left_feature, right_feature, acclerate_feature, checkpoint_feature])
        max_values = np.array([max_left_feature, max_right_feature, max_acclerate_feature, max_checkpoint_feature])
        min_values = np.array([min_left_feature, min_right_feature, min_acclerate_feature, min_checkpoint_feature])
        
        ## normalize features
        for i in range(len(features)):
            features[i] = self.normalizeFeature(features[i], max_values[i], min_values[i])
        return features


    def learn(self, weights) -> list:
        """
        IMPLEMENT YOUR LEARNING METHOD (i.e. YOUR LOCAL SEARCH ALGORITHM) HERE

        HINT: you can call self.run_episode (see controller_template.py) to evaluate a given set of weights
        :param weights: initial weights of the controller (either loaded from a file or generated randomly)
        :return: the best weights found by your learning algorithm, after the learning process is over
        """

        # num_run = 0
        curr_state = weights
        curr_score = self.run_episode(curr_state)
        best_state = curr_state
        best_score = curr_score
        T = 50
        while T > 0:
            # num_run += 1
            new_state = generate_rand(curr_state)
            assert(new_state != curr_state)
            new_score = self.run_episode(new_state)
            delta = new_score - curr_score
            # print("delta = ", delta)
            if delta > 0 or random.random() < pow(math.e, delta/T):
                curr_state = new_state
                curr_score = new_score
                if curr_score > best_score:
                    best_state = curr_state
                    best_score = curr_score
                    # print(best_score, " | iter = ", num_run)
                    # print(best_state)
            T -= 0.5
            print(curr_score, end=",\n")
        # print(best_state)
        raise Exception("Exit now")
        return best_state

    def normalizeFeature(self, value, max, min):
        return 2 * (value - min)/(max - min) - 1


def generate_rand(state):
    nstate = state.copy()
    factor = random.uniform(-0.5, 0.5)
    for i in range(len(state)):
        if random.randint(0,1) == 0:
            nstate[i] += factor

    return nstate

