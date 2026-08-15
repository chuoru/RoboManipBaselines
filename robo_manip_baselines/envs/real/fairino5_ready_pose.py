"""The FR5's ready pose, in a module that pulls in no hardware dependencies.

Both the real FR5 env and its MuJoCo counterpart start from this pose -- the
MuJoCo one so that a `--sim` run is a meaningful dry run for the real arm (UMI
replay retargets a demo as motion RELATIVE to the initial pose, so a different
starting configuration sweeps a completely different region of the workspace).

It lives here, rather than on RealFairino5DemoEnv where it is otherwise most at
home, because importing that class executes `from fairino import Robot` -- the
vendor SDK, which is not installed on machines that only train or run MuJoCo.
Importing it from the MuJoCo env made `import robo_manip_baselines.envs` fail
outright with ModuleNotFoundError on a training box. Duplicating the numbers
instead would let the two silently drift apart, which is exactly the failure
this constant exists to prevent.
"""

FAIRINO5_READY_POSE_DEG = [-65.000, -93.000, -142.000, 56.000, 64.000, -1.000]
