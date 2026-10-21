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