# Imports.
import numpy as np
import numpy.random as npr
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

from SwingyMonkey import SwingyMonkey


class Learner(object):

    def flatten_state(self, state):
        arr = []
        arr.append(state['score'])
        arr.append(state['tree']['dist'])
        arr.append(state['tree']['top'])
        arr.append(state['tree']['bot'])
        arr.append(state['monkey']['vel'])
        arr.append(state['monkey']['top'])
        arr.append(state['monkey']['bot'])

        return np.array(arr)

    def calc_gravity(self, positions):
        velocity = []
        accel = []
        for i, position in enumerate(positions):
            if i < len(positions) - 1:
                velocity.append[positions[i+1] - i]
        for i, vct in enumerate(velocity):
            if i < len(velocity) - 1:
                accel.append[velocity[i+1] - i]
        return np.average(accel)


    def build_model(self):
        print("Building model...")
        model = Sequential()
        model.add(Dense(150, init='glorot_normal', input_shape=(7,)))
        model.add(Activation('relu'))

        model.add(Dense(2, init='lecun_uniform'))
        model.add(Activation('linear'))

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)

        print("Finished building model...")
        return model

    def __init__(self):
        # learning rate
        self.eta = 0.5
        # discount rate
        self.gamma = 0.90
        # epsilon
        self.epsilon = 0.05

        self.batch_size = 10

        self.model = self.build_model()
        self.D = []

        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.runs = 0
        self.prev_epoch = 0
        self.epoch = 0

    def reset(self):
        self.runs = 0
        self.prev_epoch = self.epoch
        self.epoch += 1;

        self.last_state  = None
        self.last_action = None
        self.last_reward = None


    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # if((self.epoch != self.prev_epoch)):

        state = self.flatten_state(state)
        if(self.runs > 2):
            self.D.append([self.last_state, self.last_action, self.last_reward, state])

            batch = []
            selections = np.random.choice(len(self.D), self.batch_size)
            for val in selections:
                batch.append(self.D[val]);

            # np.random.choice(self.D, self.batch_size)
            inputs = []
            targets = []
            for i, val in enumerate(batch):
                lls = val[0]
                la = val[1]
                lr = val[2]
                ls = val[3]
                inputs.append(lls)
                prevQ = self.model.predict(np.array([lls]))[0]

                Q_val = self.model.predict(np.array([ls]))[0]
                prevQ[la] = lr + self.gamma * np.max(Q_val)
                targets.append(prevQ)

                self.model.train_on_batch(np.array(inputs), np.array(targets))

        new_action = 0
        # if 1 - epsilon, exploit
        if (npr.random() < (1 - self.epsilon)):
        # find action that has max Q(state,a)
            Q_val = self.model.predict(np.array([state]))[0]
            new_action = np.argmax(Q_val)

        # with prob epsilon, choose random action -- explore
        else:
            if (npr.random() < 0.5):
                new_action = 1
            else:
                new_action = 0
        # print(new_action)
        self.last_action = new_action
        self.last_state  = state

        self.runs += 1


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
	run_games(agent, hist, 1000, 10)

	# Save history.
	np.save('hist_nn',np.array(hist))
