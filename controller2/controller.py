import controller_template as controller_template
import math, random

S_DIST_LEFT = 0
S_DIST_FRONT = 1
S_DIST_RIGHT = 2
S_ON_TRACK = 3
S_NXT_CHKP = 4
S_VEL_CAR = 5
S_DET_ENEMY = 6
S_DIST_ENEMY = 7
S_ANGL_ENEMY = 8

class Controller(controller_template.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)
        self.prev_sensors = [0, 0, 0, 0, 0, 0, 0, 0, 0]


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
            0: track_distance_left: 1-100
            1: track_distance_center: 1-100
            2: track_distance_right: 1-100
            3: on_track: 0 or 1
            4: checkpoint_distance: 0-???
            5: car_velocity: 10-200
            6: enemy_distance: -1 or 0-???
            7: position_angle: -180 to 180
            8: enemy_detected: 0 or 1
          (see the specification file/manual for more details)
        :return: A list containing the features you defined
        """

        if self.prev_sensors[0] == 0.0: # There is no previous state
            self.prev_sensors = sensors
            # Assumes we're going in the correct direction at the beginning of the race
            return [-1, 0] # feature_collision_dist = -1

        # feature forward: high if going in the forward direction
        # feature_forward = self.prev_sensors[S_NXT_CHKP] - sensors[S_NXT_CHKP]

        # feature collision_dist: probability of possible collision
        # -1 means collision is very unlikely. 1 means collision is very likely
        feature_collision_dist = 1 - 2 * (sensors[S_DIST_FRONT]-1) / 99

        # feature collision_dir: which direction a collision is most likely
        # -1 means far left, +1 means far right, 0 means it's balanced
        # -0.3 means slightly left, +0.3 means slightly right, etc
        feature_collision_dir = (sensors[S_DIST_LEFT] - sensors[S_DIST_RIGHT]) / 99

        features = [feature_collision_dist, feature_collision_dir]
        self.prev_sensors = sensors
        return features


    def learn(self, weights) -> list:
        """
        IMPLEMENT YOUR LEARNING METHOD (i.e. YOUR LOCAL SEARCH ALGORITHM) HERE

        HINT: you can call self.run_episode (see controller_template.py) to evaluate a given set of weights
        :param weights: initial weights of the controller (either loaded from a file or generated randomly)
        :return: the best weights found by your learning algorithm, after the learning process is over
        """
        # TODO: A multistart on top of this could yield better results
        num_run = 0
        curr_state = weights
        curr_score = self.run_episode(curr_state)
        best_state = curr_state
        best_score = curr_score
        T = 25
        while T > 1:
            num_run += 1
            new_state = generate_rand(curr_state)
            new_score = self.run_episode(new_state)
            delta = new_score - curr_score
            if delta > 0 or random.random() < pow(math.e, delta/T):
                curr_state = new_state
                curr_score = new_score
                if curr_score > best_score:
                    best_state = curr_state
                    best_score = curr_score
                    print(best_score, " | iter = ", num_run)
                    print(best_state)
            T -= 0.1
        print(best_state)
        return best_state


def generate_rand(state):
    nstate = state
    factor = random.random() * 0.25
    for i in range(len(state)):
        nstate[i] *= factor * (1 if random.randint(0,1) == 0 else -1)
    
    return nstate
