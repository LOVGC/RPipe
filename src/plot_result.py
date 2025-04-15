from module import load
import os

tag = '0_MNIST_linear'
exp_path = os.path.join('output', 'result', tag)
test_result = load(exp_path, mode='torch')

print(test_result['logger']['test']['mean'])


tag = '1_MNIST_linear'
exp_path = os.path.join('output', 'result', tag)
test_result = load(exp_path, mode='torch')

print(test_result['logger']['test']['mean'])