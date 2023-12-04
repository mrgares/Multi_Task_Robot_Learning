# Multi_Task_Robot_Learning

Goal relabelling with intrinsic motivation for multi-task
autonomous robot learning

PreRequisites :

Deep Reinforcement Learning (UC Berkeley CS 285 by Prof. Sergey Levine, Chelsea Finn’s advisor)
https://www.youtube.com/playlist?list=PL_iWQOsE6TfXxKgI1GgyV1B_Xa0DxE5eH

"RND novelty rewards: https://arxiv.org/abs/1810.12894

Example of goal-conditioned autonomous robot learning: https://www.deepmind.com/blog/robocat-a-self-improving-robotic-agent"


Useful Simulations:

1. Meta World: https://meta-world.github.io/
2. Mujoco: https://mujoco.org/
3. EARL: https://architsharma97.github.io/earl_benchmark/
4. GYM: https://www.gymlibrary.dev/index.html

## Project board

The project task board can be find here: https://github.com/users/mrgares/projects/1/views/1

## Dependencies

For this project a docker container was created. Please follow these steps to setup the environment (you should be in the same path as the dockerfile):

1. Build Dockerfile

    `docker build -t robomimic:v1.0 .`

2. Create container (this assumes you want to run the project on GPU and with a DISPLAY)

    ``docker run --name robomimic_env -p 8888:8888 -p 5252:5252 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v `pwd`:/project -it --env QT_X11_NO_MITSHM=1 --device /dev/dri --privileged --gpus all --ipc=host robomimic:v1.0``

* Ports open to work with this container are set to 5151 and 5252 if you require different ports feel free to modify them.

3. Everytime we want to run container

    `docker start robomimic_env`

    `docker exec -it robomimic_env bash`