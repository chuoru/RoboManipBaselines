# Evaluation results

## Evaluation in MuJoCoUR5e environments
We performed rollouts at six different object positions for each of the five checkpoints trained with different random seeds per task, and computed the success rates.

| <nobr>Policy \\ Task</nobr> | [Cable](./environment_catalog.md#MujocoUR5eCable) | [Ring](./environment_catalog.md#MujocoUR5eRing) | [Particle](./environment_catalog.md#MujocoUR5eParticle) | [Cloth](./environment_catalog.md#MujocoUR5eCloth) | [Door](./environment_catalog.md#MujocoUR5eDoor) | [Toolbox](./environment_catalog.md#MujocoUR5eToolbox) | [CabinetSlide](./environment_catalog.md#MujocoUR5eCabinet) | [CabinetHinge](./environment_catalog.md#MujocoUR5eCabinet) | [Insert](./environment_catalog.md#MujocoUR5eInsert) | Average |
|---|---|---|---|---|---|---|---|---|---|---|
| [**ACT**](../robo_manip_baselines/policy/act) | 33<br>(&plusmn;42) | 50<br>(&plusmn;26) | 47<br>(&plusmn;30) | 77<br>(&plusmn;22) | 53<br>(&plusmn;43) | 57<br>(&plusmn;15) | 87<br>(&plusmn;14) | 13<br>(&plusmn;18) | 37<br>(&plusmn;18) | **50<br>(&plusmn;22)** |
| [**SARNN**](../robo_manip_baselines/policy/sarnn) | 80<br>(&plusmn;18) | 80<br>(&plusmn;27) | 83<br>(&plusmn;17) | 100<br>(&plusmn;0) | 63<br>(&plusmn;14) | 47<br>(&plusmn;7) | 93<br>(&plusmn;9) | 60<br>(&plusmn;25) | 37<br>(&plusmn;7) | **71<br>(&plusmn;21)** |
| [**Diffusion Policy**](../robo_manip_baselines/policy/diffusion_policy) | 27<br>(&plusmn;19) | 7<br>(&plusmn;15) | 60<br>(&plusmn;28) | 93<br>(&plusmn;15) | 60<br>(&plusmn;19) | 77<br>(&plusmn;9) | 67<br>(&plusmn;0) | 87<br>(&plusmn;14) | 33<br>(&plusmn;12) | **57<br>(&plusmn;29)** |

#### Success rate vs. number of epochs
<img width="2000" height="1200" alt="epoch" src="https://github.com/user-attachments/assets/5bcc7170-9a9f-41b2-bdd1-b709fdf0d364" />

#### Success rate vs. number of training demos
<img width="2000" height="1200" alt="datanum" src="https://github.com/user-attachments/assets/f4c603d3-a63b-41bc-8bf6-620276ba3ebb" />

## Exhibition task on the real FR5 (segmented policies, CupPickPlace)
Endurance test of `bin/RolloutSegments.py --mode exhibit` (doc/segment_dagger_checklist.md, M7): grasp a cup anywhere on the table (learned), carry it (scripted), put it on the coaster a visitor placed (learned), repeat. Pass criterion: at least 30 cycles (about 1 hour) with at most one stop for the staff.

Fill in one block per test run with the output of
`python ./misc/SummarizeSegmentResults.py ./logs/segment_results_CupPickPlace_<date>.yaml`.

### Run template
- Date / place / lighting:
- Checkpoints (grasp / place):
- Coaster positions (range, edges included?):

(paste the summary here)

Notes (failure cases, what to collect or retrain next):
