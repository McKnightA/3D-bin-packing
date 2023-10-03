import numpy as np


class ThreeDimBin:
    """
    An environment for deep reinforcement learning.
    It maintains a 3 dimensional container, with a fixed size, that will have objects packed inside.
    It provides a reward based on how well an object is placed within the container.
    """

    def __init__(self, length: int, width: int, height: int, objects):
        """
        create a new bin sized LxWxH and track the objects to be put in the bin
        :param length: the length of the bin
        :param width: the width of the bin
        :param height: the height of the bin
        :param objects: the objects to be packed
        """
        self.bin = np.zeros((length, width, height))
        self.objects = objects.copy()
        self.remaining_objects = objects.copy()

    def reset(self, objects=None):
        """
        returns the bin to an empty state
        allows the opportunity to change the objects to be packed
        :param objects: the objects to be packed
        :return: the initial state of the bin and the objects
        """
        self.bin = np.zeros_like(self.bin)

        if objects is None:
            self.remaining_objects = self.objects.copy()
        else:
            self.objects = objects.copy()
            self.remaining_objects = objects.copy()

        return self.bin, self.remaining_objects

    def step(self, action):
        """
        transition function from the current state to the new state
        :param action: a tuple of (the index of the object to be placed,
                       the (x,y,z) position of where to place the selected object,
                       the (x,y,z) rotation of how to rotate the selected object)
        :return: the resultant state, reward earned, done boolean, and potential extra info
        """
        index, position, rotation = action

        if self.remaining_objects[index] is None:
            return (self.bin, self.remaining_objects), -1, False, None

        self.remaining_objects[index] = self.rotate_object(index, rotation)

        if self.check_collision(index, position):
            self.remaining_objects[index] = self.objects[index]
            # print("collision")
            reward = -1
        else:
            self.bin = self.place_object(index, position)
            self.remaining_objects[index] = None
            reward = self.calc_reward()  # * (len(self.objects) - np.count_nonzero(self.remaining_objects))

        done = True
        for obj in self.remaining_objects:
            if obj is not None:
                done = False

        if done:
            reward = len(self.objects)

        return (self.bin, self.remaining_objects), reward, done, None

    def rotate_object(self, object_index, direction):
        """
        Rotate the object
        :param object_index: The index in the list of objects that is the object to be rotated
        :param direction: {0, 1, 2, 3} a tuple of the number of 90 degree rotations on the x, y, and z axes
        :return: Rotated object
        """
        obj = self.remaining_objects[object_index]

        # Check if the object is a numpy array
        if not isinstance(obj, np.ndarray):
            raise TypeError("The object must be a numpy array")

        # Rotate the object
        for i in range(len(direction)):
            axes = np.arange(3)
            obj = np.rot90(obj, direction[i], axes[np.where(axes != i)])

        return obj

    def check_collision(self, object_index, location):
        """
        checks if an object at a location will collide with a bin wall or anything already in the bin
        :param object_index: the index in the list of objects that is the object to be checked
        :param location: a tuple containing the x, y, z coordinate to place the object
        :return: true if there's a collision false otherwise
        """

        obj = self.remaining_objects[object_index]

        # Check if the location is within the bin
        if not (0 <= location[0] < self.bin.shape[0] and
                0 <= location[1] < self.bin.shape[1] and
                0 <= location[2] < self.bin.shape[2] and
                0 <= location[0]+obj.shape[0] <= self.bin.shape[0] and
                0 <= location[1]+obj.shape[1] <= self.bin.shape[1] and
                0 <= location[2]+obj.shape[2] <= self.bin.shape[2]):
            return True

        view = self.bin[location[0]:location[0]+obj.shape[0],
                        location[1]:location[1]+obj.shape[1],
                        location[2]:location[2]+obj.shape[2]]

        if view[obj.astype(bool)].any():
            return True
        else:
            return False

    def place_object(self, object_index, location):
        """
        places the object in the bin
        :param object_index: the index in the list of objects that is the object to be placed
        :param location: a tuple containing the x, y, z coordinate to place the object
        :return: the bin occupied with the object
        """

        new_bin = self.bin.copy()

        obj = self.remaining_objects[object_index]
        new_bin[location[0]:location[0] + obj.shape[0],
                location[1]:location[1] + obj.shape[1],
                location[2]:location[2] + obj.shape[2]] = obj

        self.remaining_objects[object_index] = None

        return new_bin

    def calc_reward(self):
        """
        Calculates the reward for the current state.
        The reward is determined as the density within the smallest bounding box.
        :return: the reward value
        """

        # Check base case empty bin
        if self.bin.max() == 0:
            return 0

        # Find the smallest bounding box that contains all the objects in the bin.
        # The bounding box is defined by its minimum and maximum coordinates.
        min_x = 0
        for i in range(self.bin.shape[0]):
            if not self.bin[i, :, :].any():
                min_x = i + 1
            else:
                break
        min_y = 0
        for i in range(self.bin.shape[1]):
            if not self.bin[:, i, :].any():
                min_y = i + 1
            else:
                break
        min_z = 0
        for i in range(self.bin.shape[2]):
            if not self.bin[:, :, i].any():
                min_z = i + 1
            else:
                break

        max_x = self.bin.shape[0]
        for i in range(self.bin.shape[0] - 1, -1, -1):
            if not self.bin[i, :, :].any():
                max_x = i
            else:
                break
        max_y = self.bin.shape[1]
        for i in range(self.bin.shape[1] - 1, -1, -1):
            if not self.bin[:, i, :].any():
                max_y = i
            else:
                break
        max_z = self.bin.shape[2]
        for i in range(self.bin.shape[2] - 1, -1, -1):
            if not self.bin[:, :, i].any():
                max_z = i
            else:
                break

        # Calculate the density within the bounding box.
        # The density is defined as the number of objects divided by the volume of the bounding box.
        density = len(self.bin.nonzero()[0]) / ((max_x - min_x) * (max_y - min_y) * (max_z - min_z))

        return density


