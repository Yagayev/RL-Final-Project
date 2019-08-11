from LunarLanding import *


def main():
    agent = LunarLandingNeuralAgent()
    env = gym.make('LunarLander-v2')
    demos = [
        # 0,
        # 50,
        # # 100,
        # 200,
        400,
        # 600,
        # 800,
        1000,
        # 1500,
        # 2000,
        # 3000,
        # 5000,
        10000
    ]
    for i in demos:
        print("After {} iterations".format(i))
        agent.network.model = load_model('./models/' + neural_network_name + str(i) + '.h5')
        agent.epsilon = 0.01

        for _ in range(2):
            play(agent, env)


if __name__ == "__main__":
    main()