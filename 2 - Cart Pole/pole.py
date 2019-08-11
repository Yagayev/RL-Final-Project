import gym
import random
import copy
import csv

spartan = True

"""
    we encountered an issue where in some cases, in the first iterations the weights begin to converge into a
    completely wrong solution. It usually starts with an average of about 15 turns, and slowly decays to 10-8 turns, and
    it doesn't seem to ever leave that range. I's worth mentioning that a completely random choice gets an average of 20 
    turns.
    
    We tried 2 approaches to bypass this issue:
    1) probabilistic: a low initial alpha, large initial epsilon, and slow epsilon decay, allowed to lower the number
    of cases where we get the bad average from the start. 
    In most cases we still get to an average of above 190 within 25000 games, but in some games we still get stuck with
    that issue.
    
    2) the "spartan solution": when we check the average, if we are below 22, we restart all the
    parameters(weights, epsilon, alpha). this means we never get stuck in the wrong convergence loop, so we can use
    a much larger alpha, and much faster epsilon decay, which allows us to get to an average over 190 turns
    within 2000-3000 games of our first "good run", so even if we get consecutive bad starts, in most cases the
    learning process is much faster.
    
"""


if spartan:
    alpha_init = 0.1
    alpha_min = 0.001
    alpha_decay = True
    alpha_decay_factor = 0.999999

    epsilon_init = 0.8
    epsilon_decay = True
    epsilon_min = 0.0001
    epsilon_decay_factor = 0.99999
else:
    alpha_init = 0.01
    alpha_min = 0.001
    alpha_decay = False
    alpha_decay_factor = 0.999999999

    epsilon_init = 0.8
    epsilon_decay = True
    epsilon_min = 0.0001
    epsilon_decay_factor = 0.999999


"""
    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
"""


alpha = alpha_init
epsilon = epsilon_init


def cart_exleration(stateA , stateB):
    return stateA[1] - stateB[1]


def pol_exleration(stateA , stateB):
    return stateA[3] - stateB[3]


# returns scalar, [a,b]X[c,d] = a*c+b*d
def vec_scalaric_mul(vec1, vec2):
    # print("multiply_vec", vec1, vec2)
    som = 0
    for i in range(0,len(vec2)):
        som += vec1[i]*vec2[i]
    return som


def mul_vec(vec1, vec2):
    ans = []
    for i in range(0, len(vec2)):
        ans.append(vec1[i] * vec2[i])
    return ans


def add_vec(vec1, vec2):
    ans = []
    for i in range(0, len(vec2)):
        ans.append(vec1[i] + vec2[i])
    return ans


def sub_vec(vec1, vec2):
    ans = []
    for i in range(0, len(vec2)):
        ans.append(vec1[i] - vec2[i])
    return ans


def abs_vec(vec):
    ans = []
    for i in range(0, len(vec)):
        ans.append(abs(vec[i]))
    return ans


def mul_vec_by_a_acalar(multiplier, state):
    res = []
    for i in range(0, len(state)):
        res.append(multiplier * state[i])
    return res


# def eval_grad(state, weights):
#     divisor = 1 / vec_scalaric_mul(state, weights)
#     multiplier = mul_vec_by_a_acalar(divisor, weights)
#     ret = mul_vec(multiplier, state)
#     # print("eval grad", state, weights, divisor, multiplier, ret)
#     return ret
    # sum of the abs of all values = 1
def normalizeOUT(vec):
        prblematic = False
        for i in range(0,len(vec)):
            if  (-0.0001 < vec[i] < 0.0001):
                prblematic = True
        else:
            normalize_to_one(vec)
        return vec

def normalize_to_one(vec):
    diviser = 0
    for v in vec:
        diviser += abs(v)
    return mul_vec_by_a_acalar(1/diviser, vec)

class EvalActions:

    def __init__(self):
        self.weights_left = [1, 1, 1, 1 ,1,1]
        self.weights_right = [1, 1, 1, 1 ,1,1]

    def eval_grade(self, state, action):
        if action == "left":
            return vec_scalaric_mul(self.weights_left, state)
        elif action == "right":
            return vec_scalaric_mul(self.weights_right, state)
        else:
            raise(NonLegalAction("not left or right"))

    def normalize3(self, relevent):
        for i in range(0, len(relevent)):
            if (-0.0001 < relevent[i] < 0.0001):
                diviser = 0
                for v in self.weights_left:
                    diviser += abs(v)
                for v in self.weights_right:
                    diviser += abs(v)
                mul_vec_by_a_acalar(2 / diviser, self.weights_left)
                mul_vec_by_a_acalar(2 / diviser, self.weights_right)

    def normalize2(self , relevent):
        for i in range(0, len(relevent)):
            if (-0.0001 < relevent[i] < 0.0001):
                normalize_to_one(self.weights_left)
                normalize_to_one(self.weights_right)
                return
            else:
                if (-500 > relevent[i] or relevent[i] > 500):
                    normalize_to_one(self.weights_left)
                    normalize_to_one(self.weights_right)
                    return

    def normalize(self,relevent):
        prblematic_small = False
        for i in range(0, len(relevent)):
            if (-0.01 < relevent[i] < 0.01  ):
                prblematic_small = True
        if (prblematic_small):
            for i in range(0, len(relevent)):
                self.weights_left[i] = self.weights_left[i] * 2
                self.weights_right[i] = self.weights_right[i] * 2
        else:
            prblematic_big = False
            for i in range(0, len(relevent)):
                if (-100 > relevent[i] or relevent[i] > 100):
                    prblematic_big = True
            if (prblematic_big):
                for i in range(0,  len(relevent)):
                    self.weights_left[i] = self.weights_left[i] / 2
                    self.weights_right[i] = self.weights_left[i] / 2

    def update(self, state, action, actual_grade ,nextval):
        # based on lecture 6 page page 14
        # print("state predict update", self.weights_left, state)
        expected_grade = self.eval_grade(state, action)
        grade_error = (actual_grade + nextval*0.001 -expected_grade)
        # if(-0.01 < grade_error <0.01 ):
        #     return
        if action == "left":
                # grad_times_error_vec = mul_vec_by_a_acalar(grade_error, gradient)
                state_times_w = mul_vec(self.weights_left, state)
                #print("+++++++++++++\n",state_times_w ,"\n", state ,"\n",self.weights_left )
                error_times_w_states_vec = mul_vec_by_a_acalar(grade_error, state_times_w)
                delta_w = mul_vec_by_a_acalar(alpha, error_times_w_states_vec)
                # print("update StateEval\n",
                #       "\nweights ", self.weights,
                #       "\nstate ", state,
                #       "\nactual_grade ", actual_grade,
                #       "\nexpected_grade ", expected_grade,
                #       "\ngrade_error ", grade_error,
                #       "\nerror_times_w_states_vec ", error_times_w_states_vec,
                #       "\ndelta_w ", delta_w,
                #       "\nadd_vec ", add_vec(self.weights, delta_w))
                # self.weights_left = normalize(add_vec(self.weights_left, delta_w))
                self.weights_left = add_vec(self.weights_left, delta_w)
                self.normalize3(self.weights_left)

        elif action == "right":
            # grad_times_error_vec = mul_vec_by_a_acalar(grade_error, gradient)
            state_times_w = mul_vec(self.weights_right, state)
            error_times_w_states_vec = mul_vec_by_a_acalar(grade_error, state_times_w)
            delta_w = mul_vec_by_a_acalar(alpha, error_times_w_states_vec)
            # print("update StateEval\n",
            #       "\nweights ", self.weights,
            #       "\nstate ", state,
            #       "\nactual_grade ", actual_grade,
            #       "\nexpected_grade ", expected_grade,
            #       "\ngrade_error ", grade_error,
            #       "\nerror_times_w_states_vec ", error_times_w_states_vec,
            #       "\ndelta_w ", delta_w,
            #       "\nadd_vec ", add_vec(self.weights, delta_w))
            self.weights_right = add_vec(self.weights_right, delta_w)
            self.normalize3(self.weights_right)
        else:
            raise (NonLegalAction("not left or right"))


class NonLegalAction(ValueError):
    pass


def act_epsilon_greedy(given_state, action_evaluator):
    global epsilon
    rand_seed = random.random()

    if rand_seed > epsilon:
        ans = act_greedy(given_state, action_evaluator)
    else:
        ans = act_random()
    if epsilon_decay:
        epsilon = max(epsilon * epsilon_decay_factor, epsilon_min)
    return ans


def act_greedy(given_state, action_evaluator):
    # action_left = state_to_action(given_state, "left")
    # action_right = state_to_action(given_state, "right")
    grade_left = action_evaluator.eval_grade(given_state, "left")
    grade_right = action_evaluator.eval_grade(given_state, "right")
    return 0 if grade_left > grade_right else 1


def state_to_action(state, action):
    ans = copy.deepcopy(state)
    if action == "left":
        ans.append(-1)
    elif action == "right":
        ans.append(1)
    return ans


def observation_to_score(observation):
    return  -abs(observation[2])


def act_random():
    ans = 0
    rand_seed = random.random()
    if rand_seed < 0.5:
        ans = 1
    return ans


def main():
    global alpha, epsilon
    action_evaluator = EvalActions()
    iteration = 0
    grad_sum = 0
    env = gym.make('CartPole-v0')
    rand = False
    for i_episode in range(5000000):
        observation = env.reset()

        grade = 0
        state = observation.tolist()
        state.append(0)
        state.append(0)
        old_state = False

        # print("state:", state)
        for t in range(200):
            if alpha_decay:
                alpha = max(alpha * alpha_decay_factor, alpha_min)
            if rand:
                env.render()
            # print(observation)
            # action = env.action_space.sample()
            # print(i_episode, " TEST!!!! ",
            #       "\nAction vec left", action_evaluator.weights_left,
            #       "\nAction vec right", action_evaluator.weights_right,
            #       "\nold_state " ,old_state)
            action = act_epsilon_greedy(state, action_evaluator)

            observation, reward, done, info = env.step(action)

            if not done:
                grade += 1
                direction = "left" if action ==0 else "right"
                nextstate = observation.tolist()
                cartex = cart_exleration(state, nextstate)
                polexl = pol_exleration(state, nextstate)
                nextstate.append(cartex)
                nextstate.append(polexl)
                if old_state:
                    score = observation_to_score(observation)
                    action = act_greedy(nextstate, action_evaluator)
                    nextval = action_evaluator.eval_grade(state, direction)
                    action_evaluator.update(old_state, direction, 1,nextval)
                old_state = state
                state = nextstate



            # print("reward:", reward)
            if done:
                direction = "left" if action == 0 else "right"
                score = observation_to_score(observation)
                action_evaluator.update(old_state, direction, 0, 0)

                iteration += 1
                grad_sum += grade
                rand = False
                if iteration % 500 == 0:

                    print(i_episode, "Avrage:", grad_sum/iteration,
                          "\nepsilon:", epsilon, "alpha:", alpha,
                          "\nAction vec left", action_evaluator.weights_left,
                          "\nAction vec right", action_evaluator.weights_right,
                          "\n")
                    f = open('resolts.csv', 'a')
                    with f:
                        writer = csv.writer(f)
                        writer.writerows([[i_episode]])
                    rand = True
                    if spartan:
                        if grad_sum/iteration < 22:
                            epsilon = epsilon_init
                            alpha = alpha_init
                            action_evaluator = EvalActions()

                    iteration = 0
                    grad_sum = 0

                break
    env.close()


class ValueFunction:

    def __init__(self):
        self.counter = 0

    def evaluate(self, observation):
        self.counter += 1
        return self.counter


if __name__ == '__main__':
    main()
