
# import torch

# # print(torch.__version__)

# print(torch.version.cuda)


# # import torch
# print("CUDA available: ", torch.cuda.is_available())

# import tensorflow as tf

# print(tf.__version__)
# print(tf.config.list_physical_devices('GPU'))



import converter


converter.convert_coco(rf"datasets\COCO\annotations","coco_converted/")