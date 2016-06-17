REinforcement learning with sarsa algorithm is applied for the given case below.

Consider a 8x8 grid and a robot that can move in one the main four
directions: N, W, E, and S. Robot starts at (1,1) (lowest left corner).
The goal with reward 100 is at (7,6). All other states have a reward of 0. Rewards are deterministic. The discount factor (\gamma) is 0.9. 
The next state is non-deterministic where with probability 0.5, you move in the intended directions and with probability 0.25 each, you move to one of its orthogonal neighbors.

Implement Sarsa learning and once learning is complete, display max_a
Q(s,a) values on the grid and the optimal action for each state.

(Bonus: Define a second goal at (1,6) of reward 100 and redo (1)---an episode
ends when either of the goals is reached.)
