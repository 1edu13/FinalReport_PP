### 5.4.2 Control Signals Evolution

The evolution of the action distribution (Steering, Gas, Brake) reveals a transition from rudimentary "bang-bang" control to a sophisticated understanding of vehicle dynamics and momentum conservation.

![Control Profile Radar Chart](B_control_radar.png)
*Figure 5: Multi-dimensional analysis of control dynamics. The Radar Chart illustrates the strategic shift from a novice, throttle-heavy approach (Red/200k) to a mature, balanced driving profile (Purple/3M). Note how the mature models maximize 'Consistency' and 'Efficiency' while significantly reducing raw throttle input.*

**Table 3: Evolution of Mean Action Values**

| Model | Throttle (Mean) | Brake (Mean) | Steering $\sigma$ (Activity) |
| :--- | :---: | :---: | :---: |
| **model_0200k** | **0.372** | -0.515 | 0.387 |
| **model_0500k** | 0.216 | -0.811 | 0.917 |
| **model_1000k** | -0.276 | -0.857 | 1.065 |
| **model_1250k** | -0.378 | -1.116 | 1.056 |
| **model_1500k** | -0.272 | -1.205 | 1.153 |
| **model_2000k** | -0.321 | -1.354 | 1.376 |
| **model_2500k** | -0.334 | -1.273 | **1.626** |
| **model_3000k** | **-0.519** | **-1.390** | 1.679 |

*Note: In the PPO continuous action space, negative output values correspond to a "do nothing" action (0.0) after clipping. A strongly negative mean indicates the agent is confident in **not** activating that pedal.*

#### Detailed Analysis of Control Strategies:

1.  **Throttle (Momentum Conservation):**
    * **Novice Phase (200k):** The model exhibits a positive mean throttle (**0.372**), implying a strategy of constant acceleration. This correlates with the high crash rate; the agent has not yet learned the relationship between excessive speed and loss of traction in corners.
    * **Expert Phase (2.5M+):** The mean throttle drops significantly into negative territory. This indicates that the converged agent has learned to **"coast"** (release the accelerator). By utilizing the car's momentum, the agent maintains high speeds with minimal energy expenditure, applying power only to exit corners or correct slides. This is a hallmark of professional racing efficiency.

2.  **Brake (The "Do No Harm" Threshold):**
    * The brake signal shows the most dramatic shift, dropping from **-0.51** to **-1.27** (and beyond in later stages).
    * This "deep negative" mean suggests the agent has learned that braking is expensive (loss of momentum) and risky. By pushing the mean far below zero, the agent minimizes the probability of accidental braking, reserving the brake pedal exclusively for sharp, decisive deceleration events at corner entries.

3.  **Steering (Activity vs. Stability):**
   * We analyze the **Standard Deviation ($\sigma$)** of the steering to measure "Activity" or responsiveness.
   * The data shows a clear **upward trend** in steering activity as the agent matures, peaking at the final models (**$\sigma \approx 1.68$ at 3M**).
   * **Interpretation:** Contrary to a passive "smooth" driver, the expert agent exhibits **high-frequency control adjustments**. This high activity indicates that the model is:
        1.  **Constantly micro-correcting** its trajectory to stay on the optimal racing line while driving at maximum velocity.
        2.  **Rapid Recovery:** The high variance reflects the agent's ability to react instantly to traction loss or touching the grass, snapping the car back to safety rather than drifting off slowly like the novice models.

