from dataset import MNIST


my_MNIST = MNIST(root='data/MNIST', split='train', process=False, transform=None)

print("done")

