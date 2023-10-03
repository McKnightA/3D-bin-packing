import torch
from model import PackingModel


# create some shape to play with
shape1 = torch.ones((4, 4, 4))
shape2 = torch.zeros((4, 4, 4))
shape2[0:2, 1, :] = 1
shape3 = torch.zeros((4, 4, 4))
shape3[1, 1:3, 1:3] = 1
shape3[2, 2:3, 1:2] = 1
shape3[3, 1:3, 0:2] = 1

bin_size = (10, 9, 11)
bin = torch.zeros(bin_size)

print("test instantiation")
model = PackingModel(num_samples=30, embedding_dim=128, num_heads=2, num_layers=2)

print("-" * 80)
print("test generating an action")
objects = [bin.unsqueeze(0).unsqueeze(0),
           shape1.unsqueeze(0).unsqueeze(0),
           shape2.unsqueeze(0).unsqueeze(0),
           shape3.unsqueeze(0).unsqueeze(0)]
action, value = model(objects)
print("index:\n", action[0].shape)
print("position:\n", [i.shape for i in action[1]])
print("rotation:\n", action[2].shape)
print("value:\n", value.shape)

