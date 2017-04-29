# Imports.
import numpy as np
import numpy.random as npr
import csv
import sys

from SwingyMonkey import SwingyMonkey

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
VELOCITY_STATES = 5
PIXELS_PER_BIN = 20
NUM_ACTIONS = 2 # DON'T CHANGE THIS
GAMMA = 0.9

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self, alg):
        # for state, we are taking into account
        # self.Q = np.array((DIST, TREETOP - MONKEYTOP, MONKEYBOTTOM - TREEBOTTOM, VEL,2))
        NUM_STATES = ((NUM_ACTIONS, SCREEN_WIDTH//PIXELS_PER_BIN + 1, SCREEN_HEIGHT//PIXELS_PER_BIN + 1, VELOCITY_STATES))
        self.Q = np.zeros(NUM_STATES)
        self.R = np.zeros(NUM_STATES)
        # learning rate
        # discount rate
        self.gamma = GAMMA
        # epsilon (for epsilon greedy, on-policy update)
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.best_score = 0
        self.iters = 0
        self.epoch = 1
        self.alg = alg
        # self.model = build_model()

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epoch += 1

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select an action and return it.
        # Return 0 to swing and 1 to jump.

        # new_action = npr.rand() < 0.1
        # new_state  = state

        # use epsilon greedy to return appropriate action
        # the action with max Q(state,a) with prob 1-epsilon
        # a random action with prob epsilon

        # if 1 - epsilon, exploit
        self.epsilon = 1/self.epoch

        HOR = int(state['tree']['dist'] // PIXELS_PER_BIN)
        VER = int((state['tree']['top'] - state['monkey']['top']) // PIXELS_PER_BIN)
        VEL = int(state['monkey']['vel'] // 20)

        if (npr.random() < (1 - self.epsilon)):
        # find action that has max Q(state,a)
            if (self.Q[1][HOR, VER, VEL] > self.Q[0][HOR, VER, VEL]):
                new_action = 1
            else:
                new_action = 0
        # with prob epsilon, choose random action -- explore
        else:
            if (npr.random() < 0.5):
                new_action = 1
            else:
                new_action = 0

        # update Q accordingly
        if (not (self.last_state is None)):
            s = self.last_state
            a = self.last_action
            r = self.last_reward
            hor = int(s['tree']['dist'] // PIXELS_PER_BIN)
            ver = int((s['tree']['top'] - self.last_state['monkey']['top']) // PIXELS_PER_BIN)
            vel = int(s['monkey']['vel'] // 20)

            # self.Q[s,a] -= self.eta*(self.Q[s,a] - (r + self.gamma * self.Q[state,new_action]))
            temp_Q = None
            if self.alg == 'qlearn':
                temp_Q = np.max(self.Q[:,HOR, VER, VEL]) 
            elif self.alg == 'sarsa':
                temp_Q = self.Q[new_action, HOR, VER, VEL]
            alpha = 1/self.R[self.last_action][hor, ver, vel]
            # Takes average
            self.Q[self.last_action][hor, ver, vel] += alpha * (self.last_reward + self.gamma * temp_Q - self.Q[self.last_action][hor, ver, vel])

        # save current action/state for next step
        self.last_action = new_action
        self.last_state  = state
        self.R[new_action][HOR, VER, VEL] += 1

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward



def run_games(learner, hist, iters = 1000, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    counter = 0
    while counter < iters:
        try:
            # Make a new monkey object.
            swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                                 text=f"Epoch {counter}: {learner.best_score}",       # Display the epoch on screen.
                                 tick_length = t_len,          # Make game ticks super fast.
                                 action_callback=learner.action_callback,
                                 reward_callback=learner.reward_callback)

            # Loop until you hit something.
            while swing.game_loop():
                pass

            # Save score history.
            if(swing.score > learner.best_score):
                learner.best_score = swing.score
            hist.append(swing.score)

            # Reset the state of the learner.
            learner.reset()
            counter += 1
        except:
            pass

    return


if __name__ == '__main__':
    ALGORITHM = sys.argv[1]
    PIXELS_PER_BIN = int(sys.argv[2])

    # Select agent.
    agent = Learner(ALGORITHM)

    # Empty list to save history.
    hist = []

    # Run games.
    run_games(agent, hist, 3000, 1)

    # Save history.
    np.save(f"{ALGORITHM}_hist{PIXELS_PER_BIN}",np.array(hist))
    with open(f"{ALGORITHM}{PIXELS_PER_BIN}.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i, data in enumerate(hist):
            writer.writerow([i, data])