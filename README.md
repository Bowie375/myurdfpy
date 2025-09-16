# myurdfpy
---

## Introduction
This is a Python library for parsing URDF (Unified Robot Description Format) files. It is based on the [yourdfpy](https://github.com/clemense/yourdfpy.git) library, but with some modifications and improvements.

## Features
TODO

## Files
TODO

## Comparison betweem urdf.py and urdf_rbt.py
1. parse URDF file:
    - urdf.py: ~0.01 sec
    - urdf_rbt.py: ~0.06 sec

2. create trimesh scene:
    this part is the same for both libraries. So I just list some results about the routine itself:
    - create scene with loading mesh: ~10 sec
    - create scene without loading mesh: ~0.002 sec
    - create scene_collision with loading mesh: ~0.33 sec
    - create scene_collision without loading mesh: ~0.002 sec

3. forward kinematics:
    - urdf.py: total time for 10 runs (averaged over 20 trys): 0.018141 sec
    - urdf_rbt.py: total time for 10 runs (averaged over 20 trys): 0.021489 sec

4. inverse kinematics:
    - urdf_rbt.py: 
        - "ik_LM" + "method='chan'" takes ~0.01ms
        - "ik_GN" takes ~0.02ms
        - "ik_NR" takes ~0.025ms