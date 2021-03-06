# Imports.
import numpy as np
import numpy.random as npr
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        # for state, we are taking into account
        # self.Q = np.array((DIST, TREETOP - MONKEYTOP, MONKEYBOTTOM - TREEBOTTOM, VEL,2))
        self.Q = np.array((NUM_STATES, 2))
        # learning rate
        self.eta = 0.5
        # discount rate
        self.gamma = 0.9
        # epsilon (for epsilon greedy, on-policy update)
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.model = build_model()

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None


    def build_model():
        print("Building model...")
        model = Sequential()
        model.add(Dense(120, init='lecun_uniform', input_shape=(7,)))
        model.add(Activation('relu'))

        model.add(Dense(180, init='lecun_uniform'))
        model.add(Activation('relu'))

        model.add(Dense(2, init='lecun_uniform'))
        model.add(Activation('linear'))

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)

        print("Finished building model...")
        return model

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
        if (npr.random() < (1 - self.epsilon)):
        # find action that has max Q(state,a)
            if (self.Q[state,1] > self.Q[state,0]):
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
            self.Q[s,a] -= self.eta*(self.Q[s,a] - (r + self.gamma * self.Q[state,new_action]))

        # save current action/state for next step
        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward



def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()

    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games.
	run_games(agent, hist, 20, 10)

	# Save history.
	np.save('hist',np.array(hist))
