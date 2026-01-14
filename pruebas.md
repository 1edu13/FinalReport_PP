#### 5.1.2 Methodology Refinement & Evaluation Protocol

Initially, the training was conducted using the standard environment configuration (`train.py`) and evaluated using default metrics. However, an in-depth analysis of these preliminary results revealed inconsistent behaviors. While the agents were often able to complete the maximum episode length (1000 steps), visual inspection via video logs showed that the driving was erratic. The agents frequently survived by spinning in circles or drifting off-track without being penalized sufficiently, exploiting the survival reward rather than learning proper lane-keeping.

**Hypothesis:**
For an autonomous vehicle, safety and road adherence must take precedence over raw velocity. We hypothesized that introducing a strict negative penalty for leaving the designated track would force the agent to learn stability and reduce "cheating" behaviors.

**Reward Policy Evolution:**
To test this hypothesis, we formally defined two distinct reward policies used during the experimentation phase.

**A. Baseline Policy (Standard CarRacing-v2)**
*Used in initial training experiments (`train.py`).*

The default reward structure focuses purely on velocity and track completion. The reward $R_t$ at step $t$ is defined as:

$$R_t = \underbrace{ \left( \frac{1000}{N} \cdot \Delta_{visited} \right) }_{\text{Progress}} - \underbrace{ 0.1 }_{\text{Time Penalty}}$$

Where:
* **Progress (+):** The agent gains $+1000/N$ points for every new track tile visited (where $N$ is the total number of tiles).
* **Time Penalty (-):** A constant cost of $-0.1$ per frame encourages speed.
* **Deficiency:** There is no explicit negative reward for driving on the grass, allowing the agent to cut corners or survive off-track.

**B. Robust Policy (Implementation: Grass Penalty)**
*Used for the final evaluated models (`train2.py`).*

To address the baseline deficiencies, we implemented a custom `GrassPenaltyWrapper` that modifies the reward structure based on visual feedback. The new reward function is:

$$R'_t = R_t - P_{grass}$$

The penalty logic ($P_{grass}$) is implemented as follows:

1.  **RGB Detection:** The wrapper analyzes the original RGB observation to detect "grass" pixels using a specific color filter (Green channel $> 150$, while Red and Blue $< 100$).
2.  **Penalty Application ($P_{grass}$):** If the ratio of green pixels in the agent's view exceeds **25%** ($green\_ratio > 0.25$), a strictly negative penalty ($-0.8$) is subtracted from the reward at each step:
    $$P_{grass} = \begin{cases} 0.8 & \text{if } green\_ratio > 0.25 \\ 0 & \text{otherwise} \end{cases}$$
3.  **Early Termination:** To prevent the agent from wandering indefinitely in the field, the episode is automatically terminated if the car remains off-track for more than **50 consecutive frames** ($max\_off\_track$).

**Evaluation Metrics:**
For the comparative analysis, we use **Mean Reward** to assess driving quality and **Win Rate** ($\%$ episodes $> 900$ points) to determine optimal racing behavior.