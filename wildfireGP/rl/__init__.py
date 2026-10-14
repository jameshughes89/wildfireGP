"""Hooks for running learned (e.g. reinforcement learning) policies on the wildfireGP simulator.

- :mod:`wildfireGP.rl.features` builds the per-cell feature matrix a learned policy consumes (numpy only).
- :mod:`wildfireGP.rl.evaluation` wraps a learned scorer as a standard wildfireGP strategy (numpy only).
- :mod:`wildfireGP.rl.env` exposes the simulator as a Gymnasium environment for training (needs the ``rl`` extra).
- :mod:`wildfireGP.rl.scenarios` defines the training scenario panels used by the environment.
"""
