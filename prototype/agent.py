"""
An agent which is basically a np array with an act(state) and update(state, action, reward) that  does
the TD0, it should be a general agent, not knowing epileptor, just an array of actions that is updated.
Should have a learning on and off parameter
- Takes the signal info (state estimation)
- use hash maps (or arrays, figure out what is better)
(when the learning mode is on)
- Receives the rewards from the environment
- update the hash map
"""
import numpy as np


class Agent: 
  """
  An initial setup for the RL agent 
  
  """

   # left to do:
   #  Q - reward_matrix initialization as in the paper , 
    # softmax selection, 
    # reward updates, 
    # and the learning-rate schedule

  def __init__(self, n_states, actions, learning_rate=0.1,temperature = 0.1, 
                        initial_value = 0.0, learning = True, seed = None):

    if not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be between 0 and 1")

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if n_states < 1 or len(actions) == 0:
        raise ValueError("Provide at least one state and action")

    self.actions = np.array(actions)
    self.temperature =temperature
    self.learning_rate = learning_rate
    self.learning = learning 
    self.random_number = np.random.default_rng(seed)


    # states and actions table 

    self.reward_matrix = np.full((n_states, len(actions)),initial_value, dtype=float,)


    def act(self, state): 
        
        """Return the index of an action selected using softmax."""

        values = self.reward_matrix[state]
       

    # firts we need to select scale the values by the temperature
        scores = (values - np.max(values)) / temperature
        weights = np.exp(scores)
        probabilities = weights/ weights.sum()

 
    # α = (α0− a∞)βcount+α∞, β ≈ e−w/ISI
   

        

    # now pick an action using those probabilities 
        action_index = self.random_number.choice(len(self.actions),p=probabilities)
        
        return int(action_index)

    


