import caffe
from caffe.proto import caffe_pb2

caffe.set_device(0)
caffe.set_mode_gpu()

dir = '/home/ubuntu/Face-emotion-recognition-master/Train/code/'

model = dir + 'FaceEmotionNet_model.prototxt'
solver_proto = dir + 'FaceEmotionNet_model_solver.prototxt'

solver = caffe.SGDSolver(solver_proto)
solver.solve()
