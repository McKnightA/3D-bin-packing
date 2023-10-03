import numpy as np
from environment import ThreeDimBin


# create some shape to play with
shapes = [np.ones((2, 2, 2)), np.ones((1, 2, 3))]

print("test instantiation")
env = ThreeDimBin(4, 4, 4, shapes)
print(env.bin)
print(env.remaining_objects, "\n")

print("-" * 80)
print("test placing")
env.bin = env.place_object(0, (0, 0, 0))
print(env.bin)
print(env.remaining_objects, "\n")

env.bin = env.place_object(1, (3, 2, 1))
print(env.bin)
print(env.remaining_objects, "\n")

print("-" * 80)
print("test resetting")
env.bin, obj = env.reset()
print(env.bin)
print(env.remaining_objects, "\n")

print("-" * 80)
print("test rotation")
env.remaining_objects[1] = env.rotate_object(1, (1, 0, 0))
env.bin = env.place_object(1, (0, 0, 0))
print(env.bin)
print(env.remaining_objects, "\n")

env.bin, obj = env.reset()
print(env.bin)
print(env.remaining_objects, "\n")

env.remaining_objects[1] = env.rotate_object(1, (0, 1, 0))
env.bin = env.place_object(1, (0, 0, 0))
print(env.bin)
print(env.remaining_objects, "\n")

env.bin, obj = env.reset()
print(env.bin)
print(env.remaining_objects, "\n")

env.remaining_objects[1] = env.rotate_object(1, (0, 0, 1))
env.bin = env.place_object(1, (0, 0, 0))
print(env.bin)
print(env.remaining_objects, "\n")

print("-" * 80)
print("test collision")
env.bin, obj = env.reset()
print(env.bin)
print(env.remaining_objects, "\n")

env.bin = env.place_object(0, (0, 0, 0))
print(env.bin)
print(env.remaining_objects, "\n")

collision = env.check_collision(1, (0, 0, 0))
print(collision)  # true

collision = env.check_collision(1, (3, 2, 1))
print(collision)  # false

collision = env.check_collision(1, (3, 3, 3))
print(collision)  # true

print("-" * 80)
print("test reward")
env.bin, obj = env.reset()
print(env.bin)
print(env.remaining_objects, "\n")

reward = env.calc_reward()
print(reward)

env.bin = env.place_object(0, (0, 0, 0))
print(env.bin)
print(env.remaining_objects, "\n")

reward = env.calc_reward()
print(reward)

env.bin = env.place_object(1, (3, 2, 1))
print(env.bin)
print(env.remaining_objects, "\n")

reward = env.calc_reward()
print(reward)

print("-" * 80)
print("test step")
env.bin, obj = env.reset()
print(env.bin)
print(env.remaining_objects, "\n")

state = env.step((1, (1, 1, 1), (1, 1, 1)))

print(state)
