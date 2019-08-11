import gym
import math
import random
import json
import csv

iterations = 3000000
print_every_iters = 100000
file = False
filename = 'test2.json'

alpha = 0.3
alpha_min = 0.000000001
alpha_decay = True
alpha_decay_factor = 0.99999999

epsilon = 0.8
epsilon_decay = True
epsilon_min = 0.000000001
epsilon_decay_factor = 0.999999

def act(given_state):
	# return act_soft_max(given_state)
	return act_epsilon_greedy(given_state)
	# return act_random()
	# return act_greedy(given_state)
Matrix = [[[[0 for x in range(2)] for y in range(2)] for z in range(22)] for q in range(22)]

stopped = {}
for i in range(23):
	stopped[i] = 0

if(file):
	try:
	    with open(filename) as data_file:
	        Matrix = json.load(data_file)
	except:
	    print("No file inisiall 0")
	finally:
	    print(Matrix)


def act_soft_max(given_state):
    first = math.exp(given_state[0])
    second = math.exp(given_state[1])
    prob = first/(first+second)
    if prob>random.random():
        return 0
    else:
        return 1

def act_epsilon_greedy(given_state):
	global epsilon
	ans = 0
	if(given_state[0] < given_state[1]):
		ans = 1
	rand_seed = random.random()

	if(rand_seed < epsilon):
		ans = (ans + 1) % 2
	if(epsilon_decay):
		epsilon = max(epsilon_min, epsilon*epsilon_decay_factor)
	return ans

def act_greedy(given_state):
	global epsilon
	ans = 0
	if(given_state[0] < given_state[1]):
		ans = 1
	return ans		

def act_random():
	ans = 0
	rand_seed = random.random()
	if(rand_seed < epsilon):
		ans = 1
	return ans




def get_stat(observ):
    me = min(observ[0], 22)
    diler = min(observ[1], 22)
    ace = 0
    if(observ[2]):
        ace = 1
    
    return Matrix[me][diler][ace]





def update_q(old_state, new_observe, chosen_action, reword, is_done):
    global alpha
    old_q = old_state[chosen_action]
    
    if is_done:
        
        old_state[chosen_action] = (1 - alpha) * old_q + alpha * reword
    else:
        new_state = get_stat(new_observe)
        best_new = max(new_state[0], new_state[1])
        old_state[chosen_action] = (1-alpha)*old_q+alpha*(reword + best_new )
    if (alpha_decay):
    	alpha = max(alpha_min, alpha*alpha_decay_factor)

env = gym.make('Blackjack-v0')


score = 0
def run(stratagy, n):
	global score
	curr_score = 0
	iters = 0
	for i_episode in range(n):
	    observation = env.reset()
	    for t in range(100):
	        # env.render()
	        # print(observation)
	        state = get_stat(observation)
	        action = stratagy(state)
	        # action = env.action_space.sample()
	        
	        observation, reward, done, info = env.step(action)
	        update_q(state, observation, action, reward, done)

	        # print(observation)
	        
	        if done:
	            score += reward
	            curr_score += reward
	            iters +=1
	            stopped_at = min(observation[0], 22)
	            stopped[stopped_at] = stopped[stopped_at]+1
	            # print("Episode finished after {} timesteps".format(t+1),observation,"reward:",reward)
	            break
	    if (iters==print_every_iters):
	        print "the", i_episode+1, "iteration, avarage:", curr_score/iters, "total avarage: ", score/(i_episode+1), "alpha:", alpha, "epsilon:", epsilon
	        curr_score = 0
	        iters = 0


run(act_epsilon_greedy, iterations)
print "total score:", score, "avarage:", score/iterations
print "stopped at:", stopped, "\n\n"
score = 0
for i in range(23):
	stopped[i] = 0 #nulifing the stopped array
print "running with greedy selaction"
run(act_greedy, print_every_iters*10)
print "total score:", score, "avarage:", score/(print_every_iters*10)
print "stopped at:", stopped
with open(filename, 'w') as outfile:
    json.dump(Matrix, outfile, indent=4)
