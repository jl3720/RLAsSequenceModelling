import numpy as np
import matplotlib.pyplot as plt


def make_prob_overlay(state, action_dist):
    fig, ax = plt.subplots(1, 1)
    x_range = range(7)
    y_range = range(6)
    color_list = ['y', 'r']
    color_list_names = ["yellow", "red"]
    if np.sum(state)<0:
        player_color = color_list_names[0]
    else:
        player_color = color_list_names[1]
    ax.set_xlim([-0.5, 6.5])
    ax.set_ylim([-0.5, 5.5])
    for x in x_range:
        for y in y_range:
            if state[y,x] != 0:
                ax.plot(x, 5-y, ".", markersize = 55,  color = color_list[np.maximum(0, state[y,x])], zorder = 5)
            else:  ax.plot(x, 5-y, ".", markersize = 55,  color = "white", zorder = 2)
            
    ax.set_xticks([])
    ax.set_ylabel("Probability of placement in columns")
    ax.set_yticks(np.linspace(-0.5, 5.5, 11) , labels = [10*i for i in range(11)])
    ax.set_facecolor("#2097f7")
    ax.bar(x_range, action_dist*6, bottom=-0.5, color = "#ffffff", zorder = 32,alpha = 0.4 ,edgecolor = "k" , linewidth = 2)    
   
    fig.suptitle(f"Probability distribution over next action for player {player_color}")
    plt.savefig("1.pdf")
    plt.show()