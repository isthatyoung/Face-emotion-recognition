import caffe
from caffe.proto import caffe_pb2

caffe.set_device(0)
caffe.set_mode_gpu()

dir = '/home/ubuntu/face-emotion-detection/train/'

model = dir + 'VGG_ILSVRC_16_layers_model.prototxt'
solver_proto = dir + 'VGG_ILSVRC_16_solver.prototxt'

solver = caffe.SGDSolver(solver_proto)
solver.solve()
