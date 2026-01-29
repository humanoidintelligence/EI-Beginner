# EI-Beginner Embodied Intelligence Introductory Practice

[中文版本](./README.md) | [English Version](./README_EN.md)

Students interested in applying to or joining the Embodied Intelligence/Humanoid Robot Intelligence team of the [OpenMOSS Lab](https://openmoss.github.io/) are required to first complete the following introductory practice (tasks 1-3 are mandatory, tasks 4-6 are optional). We share this introductory practice as a reference for all embodied intelligence/humanoid robot intelligence enthusiasts.

> Authors: [Peng Li](https://artpli.github.io) and [Siyin Wang](https://sinwang20.github.io)     
> Advised by [Prof. Xipeng Qiu](https://xpqiu.github.io)    
> Reproduction is permitted only with proper attribution.    

## Prerequisites
- Know how to use ChatGPT/DeepSeek and Google
- Know how to use Linux
- Know how to use Git and GitHub
  - https://learngitbranching.js.org/

## Task 1: Traditional Kinematics-based Robotic Arm Object Grasping
<img src="./assets/task1.png" width="50%" alt="Description">

Learn fundamental knowledge in traditional robotics, such as basic coordinate transformations, forward and inverse kinematics, dynamics, control theory, etc. Implement robotic arm object grasping based on traditional robotic motion control in PyBullet/Mujoco simulation and optionally on the laboratory's actual robotic arm (while also learning the Robot Operating System - ROS).

Notes:
- Reference Textbooks
  - Introduction to Robotics: Mechanics and Control from Stanford
  - Robotic Manipulation from MIT
- PyBullet Simulation Link: https://github.com/bulletphysics/bullet3
- Mujoco Simulation Link: https://mujoco.org/
- Challenge Game: https://rcfs.ch/

## Task 2: Reinforcement Learning-based Robotic Arm Object Grasping
<img src="./assets/task2.png" width="50%" alt="Description">

1. Learn the basics of reinforcement learning, and train and test the success rate using reinforcement learning methods on several tasks of interest in the OpenAI Gym environment.
2. Train a robotic arm object grasping policy in PyBullet/Mujoco, and attempt to deploy it on a real robot, experiencing the Sim2Real process.

Notes:
- Reference Textbooks
  - Relevant chapters of "Neural Networks and Deep Learning", Xipeng Qiu
  - Introduction to Reinforcement Learning, 2nd Edition & David Silver's UCL Course
  - UCB 285 Deep Reinforcement Learning
- OpenAI Gym Link: https://gymnasium.farama.org/index.html

## Task 3: Imitation Learning-based Robotic Arm Object Grasping
<img src="./assets/task3.png" width="50%" alt="Description">


1. Reproduce the classic imitation learning baseline method, Diffusion Policy
   - https://diffusion-policy.cs.columbia.edu
2. Simultaneously learn the Huggingface robotics learning framework, LeRobot
   - https://github.com/huggingface/lerobot

## Task 4: VLA Large Model-based Robotic Arm Object Grasping
<img src="./assets/task4.png" width="50%" alt="Description">


Learn and utilize existing VLA (Vision-Language-Action) large models such as OpenVLA/Pi/GR00T N1, etc., to explore training specialized VLA robotic arm models using existing robotic arm datasets like Open-X Embodiment.

Notes:
- OpenVLA: https://github.com/openvla/openvla
- Pi: https://github.com/Physical-Intelligence/openpi
- Gr00t: https://github.com/NVIDIA/Isaac-GR00T
- Open-X Embodiment: https://robotics-transformer-x.github.io/
- Large Models: https://stanford-cs336.github.io/spring2025/

## Task 5: LLM/VLM Large Model-based Task Planning
<img src="./assets/task5_1.png" width="50%" alt="Description">

- Desktop-level Task Planning
   - Referring to the paper "Code as Policies: Language Model Programs for Embodied Control" (ICRA 2023 Outstanding Paper, https://code-as-policies.github.io/), implement robotic arm object grasping based on a vision module and a large language model in PyBullet simulation and optionally on the laboratory's actual robotic arm.
   - Prompt existing LLMs/VLMs to complete this task.
   - Fine-tune existing LLMs/VLMs to complete this task.

<img src="./assets/task5_2.png" width="50%" alt="Description">

- Scene-level Task Planning
   - Configure any simulation environment, select a model, and successfully run the baseline.
   - Design In-Context Learning (ICL) or Chain-of-Thought (CoT) methods to improve the model's embodied planning performance.
   - (Optional) Use the training set provided by the corresponding benchmark to perform Supervised Fine-Tuning (SFT) on the model and compare the results.

  - Optional Simulation Environments/Benchmarks
    - EAI https://github.com/embodied-agent-interface/embodied-agent-interface
    - EmbodiedBench https://github.com/EmbodiedBench/EmbodiedBench
  - Reference Papers
    - Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making https://arxiv.org/pdf/2410.07166
    - VisualAgentBench: Towards Large Multimodal Models as Visual Foundation Agents https://arxiv.org/pdf/2408.06327
    - LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models https://arxiv.org/abs/2212.04088
    - ReAct: Synergizing Reasoning and Acting in Language Models https://arxiv.org/pdf/2210.03629

## Task 6: Reinforcement Learning-based Humanoid Robot Locomotion Control
<img src="./assets/task6.png" width="50%" alt="Description">

Reproduce the humanoid robot locomotion control method from OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning (https://omni.human2humanoid.com/). Learn the fundamentals of humanoid robots and the process of simulation training + Sim2Real.
1. Whole-body Teleoperation
2. Whole-body Imitation Learning

References:
- Unitree Robotics GitHub: https://github.com/unitreerobotics
- Hover: https://github.com/NVlabs/HOVER
- Underactuated Robotics from MIT

## Frontier Research
- How to Conduct Research
  - [An Opinionated Guide to ML Research](http://joschu.net/blog/opinionated-guide-ml-research.html), [John Schulman](http://joschu.net/index.html)
  - [GAMES003: Basic Literacy in Graphics and Vision Research](https://pengsida.net/games003/)
- Conferences & Journals
  - Robotics: Science Robotics, RSS, CoRL, ICRA, IROS, RLC
  - Machine Learning: ICLR, NeurIPS, ICML
  - Computer Vision: CVPR, ICCV, ECCV
  - Natural Language Processing: ACL, EMNLP, COLM
- Online Seminars
  - CMU Robotics Institute Seminar: https://www.youtube.com/@cmurobotics
  - MIT Robotics Seminar: https://www.youtube.com/@MITRoboticsSeminar
  - MIT Embodied Intelligence Seminar: https://www.youtube.com/@mitembodiedintelligence8675
  - Stanford Seminar: https://www.youtube.com/@stanfordonline

We extend our gratitude to the above-mentioned related works for open-sourcing their code!
