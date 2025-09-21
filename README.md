# myurdfpy

[![Powered by the Robotics Toolbox](https://raw.githubusercontent.com/petercorke/robotics-toolbox-python/master/.github/svg/rtb_powered.min.svg)](https://github.com/petercorke/robotics-toolbox-python)
[![Static Badge](https://img.shields.io/badge/powered_by-Viser-orange?style=flat)](https://viser.studio)
[![Static Badge](https://img.shields.io/badge/powered_by-yourdfpy-blue?style=flat)](https://github.com/clemense/yourdfpy)



## Introduction
This is a Python library for parsing URDF (Unified Robot Description Format) files. It provides two interchangeable implementations:

- `urdf.py` — built on top of [yourdfpy](https://github.com/clemense/yourdfpy.git)

- `urdf_rtb.py` — built on top of [robotics-toolbox-python](https://github.com/petercorke/robotics-toolbox-python.git)

Both versions are designed with nearly identical APIs, so switching between them requires minimal changes to your code.

### Why two versions?

Maintaining two backends is a compromise between speed, extensibility, and robustness:
- Performance
    - `urdf.py` is generally faster for URDF parsing and forward kinematics.
    - `urdf_rtb.py` tends to be faster for inverse kinematics.
    - Detailed benchmarks are provided in [this section](#comparison-of-speed-between-urdfpy-and-urdf_rtbpy).

- Community & Robustness
    - urdf_rbt.py benefits from the large user base and active development of Robotics Toolbox for Python. So I think it is more robust than `urdf.py`.
    
- Extensibility
    - From my own experience of developing with core code of robotics-toolbox-python, `urdf.py` appears more flexible and capable of handling complex URDF files (e.g., with floating or planar joints), although such files are relatively uncommon.

See the [Features](#features) section for a morea comparison and demo examples.

## Installation
You can install by running the following command:

```
git clone https://github.com/Bowie375/myurdfpy.git

cd ./myurdfpy

pip install -e .
```

See [Common Issues](#common-issues) for troubleshooting.

## Visualization
I provided a web-based visualization tool for URDF files. It is based [Viser](https://viser.studio) to help work around ssh conections. You can launch it with the following command::

```
python run/visualize_urdf.py urdf.yrdf_file=/path/to/your/urdf/file
```

## Features
1. Both support parsing URDF files:
    ```
    from myurdfpy.urdf import URDF

    urdf_model = URDF.load("/your/urdf/file")
    ``` 
    or 
    
    ```
    from myurdfpy.urdf_rbt import URDF

    urdf_model = URDF.load("/your/urdf/file")
    ```
    When `urdf_rtb.py` parses URDF files, it will automatically build ET models for the links, while `urdf.py` will build ET models dynamically when IK function is called. This is one of the reasons why `urdf.py` is faster in parsing URDF files but slightly slower in Inverse Kinematics.

2. Both support forward kinematics:
    ```
    urdf_model.update_cfg({"joint_name": joint_value})
    ```
    For more detailed usage cases, you can refer to the definition of this function.

3. Both support inverse kinematics (implemented in C):
    ```
    urdf_model.IK(relative_pose, end_link, start_link)
    ```
    For more detailed usage cases, you can refer to the definition of this function.


## Comparison of Speed between `urdf.py` and `urdf_rtb.py`
1. parse URDF file:
    - `urdf.py`: ~0.01 sec
    - `urdf_rtb.py`: ~0.06 sec

2. create trimesh scene:
    this part is the same for both libraries. So I just list some results about the routine itself:
    - create scene with loading mesh: ~10 sec
    - create scene without loading mesh: ~0.002 sec
    - create scene_collision with loading mesh: ~0.33 sec
    - create scene_collision without loading mesh: ~0.002 sec

3. forward kinematics:
    - `urdf.py`: total time for 10 runs (averaged over 20 trys): 0.018141 sec
    - `urdf_rtb.py`: total time for 10 runs (averaged over 20 trys): 0.021489 sec

4. inverse kinematics:
    - `urdf.py`: 
        - "ik_LM" + "method='chan'" takes ~0.06ms
        - "ik_GN" takes ~0.05ms
        - "ik_NR" takes ~0.05ms
    - `urdf_rtb.py`: 
        - "ik_LM" + "method='chan'" takes ~0.01ms
        - "ik_GN" takes ~0.02ms
        - "ik_NR" takes ~0.025ms

## Common Issues
1. When executing `import myurdfpy.urdf_rtb`, I got the following error:

        A module that was compiled using NumPy 1.x cannot be run in
        NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
        versions of NumPy, modules must be compiled with NumPy 2.0.
        Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

        If you are a user of the module, the easiest solution will be to
        downgrade to 'numpy<2' or try to upgrade the affected module.
        We expect that some modules will need time to support NumPy 2.

    This error is raised because robotics-toolbox-python has implemented core IK functions in C, and it is compiled with NumPy 1.x. To fix this issue, you can either downgrade NumPy to 1.x by running the following commands:
    
    ```
    pip install numpy==1.26.4,
    pip install scipy==1.11.4,
    pip install sympy==1.13.1,
    ```
    
    or you can try clone the robotics-toolbox-python repository from github and rebuild it:

    ```
    git clone https://github.com/petercorke/robotics-toolbox-python.git
    
    cd ./robotics-toolbox-python
    
    pip install -e .
    ```