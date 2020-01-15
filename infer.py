"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
import argparse

import numpy as np
import tensorflow as tf

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_CHANNEL_NUM = 3

parser = argparse.ArgumentParser(description='ImageNet Evaluation')
parser.add_argument('--engine', default='./trt.engine', type=str,
                    help='the model path')
parser.add_argument('--label-off', default=0, type=int,
                    help='label offset')
parser.add_argument('--data', default='./', type=str,
                    help='path to dataset')
parser.add_argument('--size', default=224, type=int,
                    help='input image size (default: 224)')
parser.add_argument('--log-name', type=str, default='eval',
                    help='log name')
args = parser.parse_args()

# set logging system
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=args.log_name+'.log')
logging.info(args)


class OptModel(object):
    def __init__(self, model_path):
        # load model
        with open(model_path, 'rb') as f:
            with trt.Runtime(trt.Logger(trt.Logger.ERROR)) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        # allocate buffers
        self.inputs = list()
        self.outputs = list()
        self.bindings = list()
        self.stream = cuda.Stream()
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append([host_mem, device_mem])
            else:
                self.outputs.append([host_mem, device_mem])
        self.context = self.engine.create_execution_context()

    def inference(self, input_data, batch_size):
        model_input = self.inputs[0]
        cuda.memcpy_htod_async(model_input[1], input_data, self.stream)
        self.context.execute_async(batch_size=batch_size,
                                   bindings=self.bindings,
                                   stream_handle=self.stream.handle)
        model_output = self.outputs[0]
        cuda.memcpy_dtoh_async(model_output[0], model_output[1], self.stream)
        self.stream.synchronize()
        return model_output[0]


class EvaluationTask(object):
    def __init__(self, height=224, width=224, data_path='/data', engine_file='trt.engine'):
        self.logging = None
        self.data_path = data_path
        self.height = height
        self.width = width
        # create tensorrt engine
        self.opt_model = OptModel(engine_file)
        #config evaluation
        self.label_off = 0
        self.dataset_dir = os.path.join(self.data_path, 'imagenet')
        self.test_list = os.path.join(self.data_path, 'val.list')
        self.batch_size = 1
        self.num_class = 1000


    def _mean_image_subtraction(self, image, means, num_channels):
        means = tf.broadcast_to(means, tf.shape(image))
        return image - means


    def _smallest_size_at_least(self, height, width, resize_min):
        resize_min = tf.cast(resize_min, tf.float32)
        # Convert to floats to make subsequent calculations go smoothly.
        height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
        smaller_dim = tf.minimum(height, width)
        scale_ratio = resize_min / smaller_dim
        # Convert back to ints to make heights and widths that TF ops will accept.
        new_height = tf.cast(height * scale_ratio, tf.int32)
        new_width = tf.cast(width * scale_ratio, tf.int32)
        return new_height, new_width


    def _resize_image(self, image, height, width):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
                                         align_corners=False)
        image = tf.squeeze(image, [0])
        return image


    def _aspect_preserving_resize(self, image, resize_min=256):
        shape = tf.shape(input=image)
        height, width = shape[0], shape[1]
        new_height, new_width = self._smallest_size_at_least(height, width, resize_min)
        return self._resize_image(image, new_height, new_width)


    def _central_crop(self, image, crop_height, crop_width):
        shape = tf.shape(input=image)
        height, width = shape[0], shape[1]

        amount_to_be_cropped_h = (height - crop_height)
        crop_top = amount_to_be_cropped_h // 2
        amount_to_be_cropped_w = (width - crop_width)
        crop_left = amount_to_be_cropped_w // 2
        return tf.slice(
          image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


    def _get_int_input(self, image):
        image = tf.expand_dims(image, axis=0)
        image = tf.transpose(image, perm=[0, 3, 1, 2])
        image = image / (151.06 / 127.0)
        return tf.clip_by_value(image, -128., 127.)


    def _image_preprocess(self, img_path, height, width, num_channels=3):
        img_raw = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(img_raw, channels=num_channels)
        image = self._aspect_preserving_resize(image)
        image = self._central_crop(image, self.height, self.width)
        image.set_shape([self.height, self.width, num_channels])
        image = self._mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)
        return self._get_int_input(image)


    def _getSessConfig(self):
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    allocator_type='BFC',
                                    per_process_gpu_memory_fraction=1.0)
        config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True,
                                gpu_options=gpu_options)
        return config


    def _read_list(self, file_list):
        f = open(file_list)
        img_path_list = []
        label_list = []
        for line in f.readlines():
            line = line.strip('\r\n')
            segs = line.split(' ')
            img_path_list.append(segs[0])
            label_list.append(segs[1])
        f.close()
        return img_path_list, label_list


    def evaluate(self):
        # load the image path list and lebel list
        img_path_list, label_list = self._read_list(self.test_list)
        img_num = len(img_path_list)

        # image preprocess, and save as cached data_array
        data_size = _CHANNEL_NUM*self.width*self.height
        data_array = cuda.pagelocked_empty(img_num*data_size, np.int8)

        config = self._getSessConfig()
        sess = tf.Session(config=config)
        with tf.name_scope('imagenet', 'preprocess'):
            image = tf.placeholder(tf.string)
            image_preprocess = self._image_preprocess(image, self.height, self.width, _CHANNEL_NUM)

        for i in range(img_num):
            #----- read and pre-process batch data ------
            print('Pre-process the image: %s' % i)
            img_path = os.path.join(self.data_path, img_path_list[i])
            batch_data = sess.run(image_preprocess, feed_dict={image: img_path})
            batch_data = batch_data.astype(np.int8).reshape(data_size)
            data_array[i*data_size:(i+1)*data_size] = batch_data

        # CNN inference for batch=1
        time_array = np.zeros(img_num)
        out_array = np.zeros((img_num, self.num_class))
        for i in range(img_num):
            #------------ process batch data ------------
            start_time = time.time()
            out = self.opt_model.inference(data_array[i*data_size:(i+1)*data_size], self.batch_size)
            time_array[i] = time.time() - start_time
            print('Infer time: %s seconds' % time_array[i])
            out_array[i] = out

        # post-process for top-k accuracy
        top_k = 5
        topk_num = 0
        for i in range(img_num):
            #------------ top-k ------------
            gt_label = int(label_list[i])
            pred_topk = np.argsort(-out_array[i])[:top_k] + self.label_off
            if gt_label in pred_topk:
                topk_num += 1

        time_mean = np.mean(time_array[10:])
        print('Mean time per sample: %s seconds' % time_mean)
        topk_acc = float(topk_num) / img_num
        print('Top-5 validation acc.: %s' % topk_acc)

        metric = dict()
        metric['top5_acc'] = float(topk_acc)
        self.logging.info(metric)
        self.logging.info(time_mean)

        return


if __name__ == '__main__':
  # create evaluation task, then obtain the results
  task = EvaluationTask(args.size, args.size, args.data, args.engine)
  task.logging = logging
  task.label_off = args.label_off
  task.evaluate()
