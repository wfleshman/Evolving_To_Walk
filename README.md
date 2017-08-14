# Learning to Walk using Evolution
![Demo](/imgs/walk.gif)

## Dependencies
````
gym version 0.9.2
numpy 1.13.1
matplotlib 2.0.0
````

## Evolutionary Algorithms
From [wikipedia](https://en.wikipedia.org/wiki/Evolutionary_algorithm):
an evolutionary algorithm (EA) is a subset of evolutionary computation, a generic population-based metaheuristic optimization algorithm. An EA uses mechanisms inspired by biological evolution, such as reproduction, mutation, recombination, and selection. 

## BipedalWalker-v2
[OpenAI](https://gym.openai.com/envs/BipedalWalker-v2)

This is an environment from OpenAI gym. The goal is for the robot to learn to walk. Reward is given to the robot for moving forward. If it falls it gets a reward of -100, additionally applying torque to the 4 motors also costs a small amount. 

The robot's state (what it can observe) consists of the hull angle speed, angular velocity, horizontal speed, vertical speed, relative position of joints, joints angular speed, leg contact with ground, and 10 lidar range measurements.

At each timestep the robot should consider its state and decide on an action. The robot can control 4 motors, sending each a signal from the continuous range of [-1,1]. 

## Genetic Algorithm
In order to leverage evolution we need a population of robots. Each robot is given a brain, which in this case is an artificial neural network. The neural networks take in a vector of inputs (the 24 sensor measurements) and output a vector corresponding to the amount of torque to apply to each of the 4 motors. Each robot's goal in life is to move forward as efficiently as possible, but not all brains are created equal, and the robots will have varying success.

Genetic algorithms are all about survival of the fittest. Therefore, we assign a fitness to each robot. The fitness is the robot's average cumulative reward on three trials of its task. We use three trials to more accurately measure a robot's ability (some will occaisonally get lucky). 

After each robot in the generation has been scored, the top 25% move onto the next generation (they survive). To fill the remaining 75% we use a form of reproduction. Again, inspired by biology, the fittest members of the population are chosen to reproduce. The previous generation's scores are converted to probabilities via a softmax operation, then parents are selected (with replacement) via this probability distribution. 

Each pair of parents produces a new child (robot sex). The child will have a neural network with the same size as the parents, but the weights of each neuron will be randomly assigned from one of the parents. Additionally, a small amount of the neurons (10%) will be mutated by adding a normally distributed amount of noise.

This process repeats itself over several generations, allowing the robot species to improve on their ability to walk (and overthrow humanity).

## Additional Info
Evolutionary algorithms are a viable solution for training neural network systems for several reasons. Neural networks are highly parameterized, and successfully optimizing the weights of the network with backprop can be slow. Success is also highly dependent on the bucket of hyperparameters that have to be hand picked by the engineer. Using evolution rids us of many parameters and often leads to great results. Traditional methods of training a network require small incremental improvements. With evolution, improvements can be significant. Furthermore, the process of calculating the population's fitness can be parallelized over several clusters, making EAs very fast!