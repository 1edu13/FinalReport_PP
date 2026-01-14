## 5.4 Stability and Control Signals

Analyzing the internal telemetry of the agent reveals how its driving strategy matures over time. We focus on two key aspects: the **stability of the reward** (reliability) and the **evolution of control inputs** (steering, throttle, and brake).

### 5.4.1 Reward Stability Analysis

Stability is measured by the **Standard Deviation ($\sigma$)** of the total reward across evaluation episodes. A lower $\sigma$ indicates a more predictable and robust driver.

![Reward Stability Box Plot](model_comparison_boxplot.png)
*Figure 4: Distribution of rewards across training checkpoints. The "height" of each box represents the Interquartile Range (IQR). Note how the 2.5M model exhibits the most compact distribution among the high-performing agents, indicating superior consistency.*

* **The "Nervous" Phase (1.0M - 2.0M steps):**
    The model at **1.0M steps** shows a high deviation ($\sigma \approx 215$), which peaks at **1.5M steps** ($\sigma \approx 295$). During this phase, the agent is capable of high scores but frequently commits critical errors, leading to a wide spread of results (long "whiskers" in the boxplot).

* **The "Reliable" Phase (2.5M steps):**
    At **2.5M steps**, we observe a drastic drop in standard deviation to **$\sigma = 144.28$**, the lowest among the high-performing models. This confirms that the agent has consolidated its policy, eliminating most catastrophic failures. Although the 2.0M model achieved a higher peak win rate (70%), it was significantly less stable ($\sigma = 235.34$), making the **2.5M model the superior candidate for deployment** due to its consistency.

### 5.4.2 Control Signals Evolution

The average values of the actions taken by the agent (Steering, Gas, Brake) tell a compelling story about energy efficiency and control confidence.

![Control Profile Radar Chart](B_control_radar.png)
*Figure 5: Radar chart comparing the normalized control profiles. The 200k model (Red) shows a bias towards Throttle, while the mature 2.5M model (Blue) expands towards Efficiency and Brake usage.*

| Model (Steps) | Throttle (Mean) | Brake (Mean) | Interpretation |
| :--- | :--- | :--- | :--- |
| **200k** | **0.372** | -0.515 | **Constant Acceleration:** The novice agent holds the gas down, leading to loss of control. |
| **1.0M** | -0.276 | -0.857 | **Learning to Let Go:** The agent starts to release the gas pedal more often. |
| **2.5M** | **-0.334** | **-1.273** | **Controlled Coasting:** The expert agent relies on momentum. |

*Note: In the PPO continuous action space, negative output values correspond to a "do nothing" action (0.0) after clipping. A strongly negative mean indicates the agent is confident in **not** activating that pedal.*

1.  **Throttle (Gas):**
    * Early models (**200k**) exhibit a **positive mean throttle (0.37)**, indicating the agent is constantly pressing the accelerator. This explains the erratic behavior and frequent off-track excursions.
    * As training progresses, the mean throttle drops significantly, becoming negative (**-0.33 at 2.5M**). This implies the converged agent has learned to **"coast"** (release the gas) for large sections of the track, applying power only when necessary to maintain speed or exit corners. This is a hallmark of smooth, professional racing lines versus the "floor-it" strategy of a beginner.

2.  **Brake:**
    * The brake signal remains deeply negative across all models (dropping from -0.51 to -1.39). The increasingly negative value at 2.5M (-1.27) suggests the agent has learned to keep the brake pedal "far from activation" to avoid accidental braking due to exploration noise, applying it only in sharp, decisive bursts.

3.  **Steering:**
    * The mean steering value remains close to zero (e.g., **-0.05 at 2.5M**), which is expected for a circuit with a balance of left and right turns. The low magnitude suggests the agent stays centered and avoids excessive zig-zagging corrections, further contributing to the stability observed in section 5.4.1.

### 5.4.3 Visual Analytics Methodology

To ensure a rigorous interpretation of the graphical data presented in Figure 4 (Boxplots) and Figure 5 (Radar Chart), we define the calculated metrics and statistical elements as follows, based on the evaluation of 30 episodes per model.

**A. Boxplot Statistics (Stability Analysis)**
The reward distribution is visualized using standard statistical boxplots (Figure 4). The elements are defined as follows:
* **The Box (Interquartile Range - IQR):** Represents the central 50% of the data, spanning from the **1st Quartile ($Q1$, 25th percentile)** to the **3rd Quartile ($Q3$, 75th percentile)**. A vertically shorter box indicates high clustering of results, synonymous with high reliability.
* **The Whiskers:** Extend to the most extreme data points within the range of $1.5 \times IQR$ from the box edges. They visualize the expected variability of the model.
* **Points (Outliers):** Individual episodes falling outside the whiskers. These represent anomalies—rare failures or exceptionally lucky runs.

**B. Control Radar Derived Metrics (Control Analysis)**
The Control Radar (Figure 5) visualizes the agent's driving "personality" by normalizing raw telemetry data. In addition to raw control inputs (Throttle, Brake), we introduce two derived metrics:
* **Efficiency ($\eta$):** Calculated as the ratio of reward to duration ($\eta = \frac{\text{Mean Reward}}{\text{Avg Steps}}$). This metric penalizes "slow and safe" driving. High efficiency indicates the agent maximizes points per second, finding optimal racing lines rather than merely surviving.
* **Consistency ($C$):** Defined as the inverse of the standard deviation ($C = \frac{1}{\sigma + \epsilon}$). This metric rewards reproducibility; a high consistency score implies the agent's performance is almost identical across all test episodes, minimizing the variance caused by random initialization or noise.
![alt text](image.png)