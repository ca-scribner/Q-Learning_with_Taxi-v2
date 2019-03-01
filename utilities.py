import numpy as np
from collections import deque
import copy


def egreedy(q, epsilon=0):
    """
    Return index and q value of the action chosen from q vector using the epsilon-greedy strategy

    This strategy returns an exploration-focused random action epsilon of the time, and an 
    exploitation-focused best action (action resulting in max Q) 1-epsilon 

    :param q: Numpy vector of Q values for each available action
    :param epsilon: Epsilon for e-greedy
    :return: Tuple of (index, q value) for the chosen action
    """
    q = np.asarray(q)
    if q.shape[0] == 0:
        raise ValueError(f"Invalid q vector {q}")
    
    # Find the greedy action.  For ties, choose one at random.
    i_greedy = np.argwhere(q == np.max(q)).reshape(-1)
    if i_greedy.shape[0] == 1:
        i_greedy = i_greedy[0]
    elif i_greedy.shape[0] > 1:
        i_greedy = np.random.choice(i_greedy)
    else:
        raise ValueError("No greedy index found.")

    # Build probability distribution
    p = np.zeros_like(q)
    p[:] = epsilon / float(q.shape[0])
    # Increment the best q's probability
    p[i_greedy] += 1 - epsilon

    # Make our choice and return
    i = np.random.choice(range(len(q)), p=p)
    return i, q[i]


def exploration(c):
    """
    Return an action according to the Exploration-First strategy, which always explores the least-visited action (min c)

    :param c: Numpy vector of times each (state,action) has been explored
    :return: Index of the action chosen
    """
    candidates = np.argwhere(c == np.min(c)).reshape(-1)
    return np.random.choice(candidates)


def init_q(n_states, n_actions, ini=0.0):
    """
    Returns an initialized Q array with n_states states (rows) and n_actions actions (columns)

    :param n_states: Number of states in MDP
    :param n_actions: Number of actions per state (uniform across all states)
    :param ini: Initialization value for all states
    :return: Numpy 2D array of Q initialized to ini
    """
    a = np.zeros((n_states, n_actions))
    a[:] = ini
    return a


def run_episode(env, agent, imax=1000, updateQ=True):
    """
    Run an episode through the environment env using given agent

    :param env: Initialized environment (environment will automatically be reset at start of episode)
    :param agent: Agent object
    :param imax: Maximum number of steps to take in environment
    :param updateQ: Boolean controlling whether q is updated
    :return: Reward generated during this episode
    """
    s = env.reset()
    r_total = 0
    for i in range(imax):
        # Figure out what to do
        i_action = agent.get_next_action(s)

        # Step in environment
        s_prime, reward, gameEnded, _ = env.step(i_action)
        r_total += reward

        # Fold event back into agent
        if updateQ:
            agent.step(s, s_prime, reward, i_action)

        if gameEnded:
            break
        else:
            s = s_prime
    return r_total


def run_episodes(env, agent, imax=10000, report_frequency=100, window=100, i_greedy=100):
    """
    Run multiple epsodes through an environment using a given agent, evaluating it along the way.

    :param env: Initialized environment (environment will automatically be reset at start of episode)
    :param agent: Agent object
    :param imax: Maximum number of episodes (plays) of the environment
    :param report_frequency: Number of iterations between statistics reporting
    :param window: Number of exploration evaluations to use for reporting recent results
    :param i_greedy: Number of evaluations of the greedy policy for each reporting interval
    :return: Report Dict containing: 
        i: List of iteration number that all other values correspond to
        r_greedy: List of mean greedy-policy reward evaluations
        r_greedy_std: List of standard deviation of greedy-policy reward evaluations
        r_greedy_min: List of min deviation of greedy-policy reward evaluations
        r_greedy_max: List of max of greedy-policy reward evaluations
        r_window: List of most recent mean rewards obtained during exploration
        dq_max_window: List of maximum change to q in the most recent explorations
    """

    report = {
        'i': [],
        'r_greedy': [],
        'r_greedy_std': [],
        'r_greedy_min': [],
        'r_greedy_max': [],
        f'r_window': [],
        'dq_max_window': [],
    }

    rewards_in_window = deque(maxlen=window)
    q_in_window = deque(maxlen=window)

    for i in range(imax):
        # Run the episode
        this_reward = run_episode(env, agent)

        # Maintain the recent rewards/q's
        if len(rewards_in_window) == window:
            rewards_in_window.popleft()
            q_in_window.popleft()
        rewards_in_window.append(this_reward)
        q_in_window.append(agent.q.copy())

        # Report results
        if i % report_frequency == 0:
            report['i'].append(i)
            report['r_window'].append(np.mean(rewards_in_window))
            report['dq_max_window'].append(abs(q_in_window[-1] - q_in_window[0]).max())

            agent_greedy = copy.deepcopy(agent)
            agent_greedy.strategy = 'egreedy'
            agent_greedy.epsilon = 0.0
            r_greedy_history = np.zeros((i_greedy))
            for j in range(r_greedy_history.shape[0]):
                r_greedy_history[j] = run_episode(env, agent_greedy, updateQ=False)
                # r_greedy = (r_greedy * (j) + this_r) / float(j+1)
            report['r_greedy'].append(r_greedy_history.mean())
            report['r_greedy_std'].append(r_greedy_history.std())
            report['r_greedy_min'].append(r_greedy_history.min())
            report['r_greedy_max'].append(r_greedy_history.max())
            print(f'{i}: r_recent = {report["r_window"][-1]:.2f}, r_greedy = {report["r_greedy"][-1]:.2f}, ' + \
                  f'dq_max_recent = {report["dq_max_window"][-1]:.4f}')
    return report

class Agent(object):
    """
    Agent class for learning environments
    """
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, strategy='egreedy', epsilon=None):
        """
        Initialize the agent with number of states, actions, and search/learning settings
        
        :param n_states: Number of possible states in the environment
        :param n_actions: Number of possible actions in the environment
        :param alpha: Learning rate applied to each q update
        :param gamma: Discount rate applied to future rewards
        :param strategy: Search strategy (eg: epsilon-greedy (egreedy) or exploration-first (exploration))
        :param epsilon: Search parameter for epsilon-greedy strategy
        """
        self._strategy = None
        self.strategy = strategy
        self.init_strategy()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = init_q(n_states=n_states, n_actions=n_actions)
        self.c = np.zeros_like(self.q)

    def init_strategy(self):
        if self.strategy == 'egreedy':
            self.get_next_action = self.get_next_action_egreedy
        elif self.strategy == "exploration":
            self.get_next_action = self.get_next_action_exploration

    def get_next_action_egreedy(self, s):
        i_action, this_q = egreedy(self.q[s, :], self.epsilon)
        return i_action

    def get_next_action_exploration(self, s):
        return exploration(self.c[s, :])

    def step(self, s, s_prime, reward, i_action):
        """
        Learn from a step in the environment, updating q and c.

        :param s: Start state for the step
        :param s_prime: End state for the step 
        :param reward: Reward obtained for the action
        :param i_action: Index of action taken
        """
        self.q[s, i_action] += self.alpha * (reward + self.gamma * self.q[s_prime].max() - self.q[s, i_action])
        self.c[s, i_action] += 1

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy
        if self.strategy == 'egreedy':
            self.get_next_action = self.get_next_action_egreedy
        elif self.strategy == "exploration":
            self.get_next_action = self.get_next_action_exploration
        else:
            raise ValueError(f"Invalid strategy {strategy}")
