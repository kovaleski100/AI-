import controller_template as controller_template
import matplotlib.pyplot as plt
import numpy as np

class Controller(controller_template.Controller):

    NUM_FEATURES = 3
    NUM_THETAS = (1 + NUM_FEATURES) * 5 # numero de parametros pra aprender
    NUM_SAMPLES = 100 # o número de amostras
    
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

        features = np.insert(self.compute_features(self.sensors), 0, 1) # insert 1 in begin
        gen_feature = np.array([*range(len(features))]) # [0, 1, 2, ... N_features]
        score_list = []
        for action in range(5):
            gen_param = np.array([*range(int(action*self.NUM_THETAS/5), int((action+1)*self.NUM_THETAS/5))])
            score_list.append(np.sum(parameters[gen_param] * features[gen_feature]))
        return np.argsort(score_list)[-1] + 1


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
        max_checkpoint_feature = 100 # TODO valores variáveis
        min_checkpoint_feature = -100 # TODO
        diff = checkpoint_distance - self.last_checkpoit_distance
        checkpoint_feature = max(min_checkpoint_feature, min(max_checkpoint_feature, diff))
        self.last_checkpoit_distance = checkpoint_distance
        
        #####
        features = np.array([left_feature, right_feature, acclerate_feature])
        max_values = np.array([max_left_feature, max_right_feature, max_acclerate_feature])
        min_values = np.array([min_left_feature, min_right_feature, min_acclerate_feature])
        
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
        cov = np.identity(self.NUM_THETAS)
        min_delta_to_stop = 0.01
        new_weights = weights
        while True:
            old_weights = new_weights
            candidates_list = self.generate(new_weights, cov, self.NUM_SAMPLES)
            # Gerou candidatos
            order = self.aval(candidates_list)
            best_thetas = candidates_list[order[-self.NUM_THETAS:]]
            # Avaliou os candidatos
            cov = self.new_cov(new_weights, best_thetas)
            # Calculou nova covariancia
            new_weights = np.mean(best_thetas[-2:], axis=0)
            # Tirou a média deles
            delta = abs(np.sum(np.power(new_weights, 2)) - np.sum(np.power(old_weights, 2)))
            if (delta < min_delta_to_stop):
                break
        
        return new_weights

    def aval(self, candidates_list):
        # TODO melhorar isso
        scores = []
        for candidate in np.rollaxis(candidates_list, 0):
            score = self.run_episode(candidate)
            scores.append(score)
        order = np.argsort(np.array(scores))
        return order

    def generate(self, mean, cov, k = 1):
        return np.random.multivariate_normal(mean, cov, k)

    def new_cov(self, mean, best_thetas):
        return (1/self.NUM_THETAS) * (np.array(best_thetas-mean)*np.array(best_thetas-mean).T)

    def normalizeFeature(self, value, max, min):
        return 2 * (value - min)/(max - min) - 1
