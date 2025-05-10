import tensorflow as tf

# 查看 TensorFlow 版本号
print("TensorFlow version:", tf.__version__)

# 检查是否支持 CUDA
print("CUDA available:", tf.test.is_built_with_cuda())

# 检查是否检测到 GPU 设备
print("GPU devices:", tf.config.list_physical_devices('GPU'))
