# Project 3: Collaboration and Competition


## About the environment

In this environment, each of the two agents controls its own racket to bounce a ball over the net. If an agent hit the ball over the net, it receives a reward of +0.1, if it misses the ball or hit the ball out of the bound, it receives a reward of -0.01. Thus the goal of the agent is to keep hitting the ball over the net without letting it go out of the bound.

The observation space is continuous, and represented with a vector of length 8, corresponding to the velocity and position of the ball and the racket, each agent receives its own local observation.

The action space is also continuous, and represented with a vector of 2, corresponding to the movement perpendicular to net and jumping.

The environment is considered solved if it achieves an average score at least +0.5 over 100 episodes, where the score of each episode is defined as the maximum of the rewards received by every agent.


## Setup the environment

TODO: The README has instructions for installing dependencies or downloading needed files.


## Instructions

TODO: The README describes how to run the code in the repository, to train the agent.