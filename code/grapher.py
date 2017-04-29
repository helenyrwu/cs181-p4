import numpy as np
from matplotlib import pyplot as plt

q20 = np.load('qlearn_hist20.npy')
s20 = np.load('sarsa_hist20.npy')
q50 = np.load('qlearn_hist50.npy')
s50 = np.load('sarsa_hist50.npy')
q20_means = [np.mean(x) for x in np.split(q20, 100)]
q50_means = [np.mean(x) for x in np.split(q50, 100)]
s20_means = [np.mean(x) for x in np.split(s20, 100)]
s50_means = [np.mean(x) for x in np.split(s50, 100)]
q20_err = [np.std(x) for x in np.split(q20, 100)]
q50_err = [np.std(x) for x in np.split(q50, 100)]
s20_err = [np.std(x) for x in np.split(s20, 100)]
s50_err = [np.std(x) for x in np.split(s50, 100)]

fig = plt.figure()

ax1 = fig.add_subplot(221)
ax1.plot(list(range(1, len(q20) + 1)), q20, 'ro', c='#F44336')
ax1.set_title('Q-Learn 20')

ax2 = fig.add_subplot(222)
ax2.plot(list(range(1, len(s20) + 1)), s20, 'ro', c='#2196F3')
ax2.set_title('Sarsa 20')

ax3 = fig.add_subplot(223)
ax3.plot(list(range(1, len(q50) + 1)), q50, 'ro', c='#F44336')
ax3.set_title('Q-Learn 50')

ax4 = fig.add_subplot(224)
ax4.plot(list(range(1, len(s50) + 1)), s50, 'ro', c='#2196F3')
ax4.set_title('Sarsa 50')

axes = [ax1, ax2, ax3, ax4]

for axis in axes:
  axis.set_xlabel('Epoch')
  axis.set_ylabel('Score')

fig3 = plt.figure()

ax11 = fig3.add_subplot(221)
ax11.errorbar(list(range(1, len(q20) + 1, 30)), q20_means, c='#F44336')
ax11.set_title('Q-Learn 20')

ax21 = fig3.add_subplot(222)
ax21.errorbar(list(range(1, len(s20) + 1, 30)), s20_means, c='#2196F3')
ax21.set_title('Sarsa 20')

ax31 = fig3.add_subplot(223)
ax31.errorbar(list(range(1, len(q50) + 1, 30)), q50_means, c='#F44336')
ax31.set_title('Q-Learn 50')

ax41 = fig3.add_subplot(224)
ax41.errorbar(list(range(1, len(s50) + 1, 30)), s50_means, c='#2196F3')
ax41.set_title('Sarsa 50')

axes = [ax11, ax21, ax31, ax41]

for axis in axes:
  axis.set_xlabel('Epoch')
  axis.set_ylabel('Average Score')

# objects = ('Q-Learn 20', 'Sarsa 20')
# hi_scores = [max(q20), max(s20)]

# y_pos = np.arange(len(objects))
# plt.bar(y_pos, hi_scores, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)

# data to plot
n_groups = 2
hi_qlearn = (max(q20), max(q50))
hi_sarsa = (max(s20), max(s50))
 
# create plot
fig2, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 1
 
rects1 = plt.bar(index, hi_qlearn, bar_width,
                 alpha=opacity,
                 color='#F44336',
                 label='Q-Learn')
 
rects2 = plt.bar(index + bar_width, hi_sarsa, bar_width,
                 alpha=opacity,
                 color='#2196F3',
                 label='Sarsa')
 
plt.xlabel('Pixels Per Bin')
plt.ylabel('High Score')
plt.title('High Scores')
plt.xticks(index + (bar_width/2), ('20', '50'))
plt.legend()



# n_groups = 2
# mean_qlearn = (np.mean(q20), np.mean(q50))
# mean_sarsa = (np.mean(s20), np.mean(s50))
# std_qlearn = (np.std(q20), np.std(q50))
# std_sarsa = (np.std(s20), np.std(s50))
 
# # create plot
# fig2, ax = plt.subplots()
# index = np.arange(n_groups)
# bar_width = 0.35
# opacity = 1
 
# rects1 = plt.bar(index, mean_qlearn, bar_width,
#                  alpha=opacity,
#                  color='#F44336',
#                  label='Q-Learn')
 
# rects2 = plt.bar(index + bar_width, mean_sarsa, bar_width,
#                  alpha=opacity,
#                  color='#2196F3',
#                  label='Sarsa')
 
# plt.xlabel('Pixels Per Bin')
# plt.ylabel('Mean Score')
# plt.title('Mean Scores')
# plt.xticks(index + (bar_width/2), ('20', '50'))
# plt.legend()

plt.show()