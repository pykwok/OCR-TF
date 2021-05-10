import tensorflow as tf
import os
import numpy as np
from tensorflow.python.ops import array_ops
import string
from distutils.version import LooseVersion
from dataprocessing import get_img_lab_files

# TensorFlow的版本判断标志
isNewAPI = True

if LooseVersion(tf.__version__) > LooseVersion("1.14"):
    print("new version API")
else:
    print('old version API')
    isNewAPI = False


# 制作数据集
def get_dataset(config, shuffle=True, repeat=True):
    img_lab_files = get_img_lab_files(config['gt_path'], config['image_path'])
    print(img_lab_files[:3])

    # 定义图像解码函数
    def _parse_function(filename, config):
        image_string = tf.io.read_file(filename[0])
        if isNewAPI == True:  # 适用于TensorFLow 1.13, 2.0及之后的版本
            img = tf.io.decode_jpeg(image_string, channels=config['ch'])
        else:  # 适用于 1.12及之前的版本
            img = tf.image.decode_jpeg(image_string, channels=config['ch'])

        # 获取图片形状
        h, w, c = array_ops.unstack(array_ops.shape(img), 3)
        img = tf.reshape(img, [h, w, c])

        # 计算输出图片在高和宽两个方向的变化率，并取出最小的那个
        zoom_ratio = tf.cond(
            tf.less(config['tagsize'][0] / h, config['tagsize'][1] / w),
            lambda: tf.cast(config['tagsize'][0] / h, tf.float32),
            lambda: tf.cast(config['tagsize'][1] / w, tf.float32)
        )
        # 在高和宽两个方向上，以变化率最小的方向为主,计算图片按照目标尺寸等比例缩放后的边长
        resize_h, resize_w = tf.cond(
            tf.less(config['tagsize'][0] / h, config['tagsize'][1] / w),
            lambda: (config['tagsize'][0], tf.cast(tf.cast(w, tf.float32) * zoom_ratio, tf.int32)),
            lambda: (tf.cast(tf.cast(h, tf.float32) * zoom_ratio, tf.int32), config['tagsize'][1])
        )

        # resize图片
        if isNewAPI == True:
            img = tf.image.resize(img, [resize_h, resize_w], tf.image.ResizeMethod.BILINEAR)
        else:
            img = tf.image.resize_images(img, [resize_h, resize_w], tf.image.ResizeMethod.BILINEAR)

        # 打开文件，读取标注
        # ground_truth = tf.io.read_file(filename[1])
        print(img.shape)

        # 对图片的像素进行归一化
        # img = img / 255.0
        img = img / 127.5 - 1

        # 定义函数，用于制作标签
        def my_py_func(filename):
            if isNewAPI == True:  # 新的API里传入的是张量，需要进行转换
                filename = filename.numpy()

            # 打开文件，读取样本的标注文件
            filestr = open(filename, "r", encoding='utf-8')
            ground_truth = filestr.readlines()
            filestr.close()

            # 将字符转换成索引
            # unk = charoffset - 1 ；填充pad = charoffset - 2
            y = [config['characters'].find(c) + config['charoffset'] for c in ground_truth[0]]
            # 防止越界
            y = y[:config['label_len']]

            # 计算“需要填充”的标签长度
            padsize = config['label_len'] - len(y)
            # 按照指定长度进行“填充”
            y.extend([config['charoffset'] - 2] * padsize)

            # 将数组封装成张量
            ground_truth = tf.stack(y)

            return tf.cast(ground_truth, dtype=tf.float32)  # 返回结果

        # 构建自动图，处理样本标签
        if isNewAPI == True:
            threefun = tf.py_function
        else:
            threefun = tf.py_func

        # 返回值类型要与函数的格式严格对应
        # my_py_func()函数：用于制作标签
        ground_truth = threefun(my_py_func, [filename[1]], tf.float32)
        # 返回 处理过的样本图片和标签
        return img, ground_truth

    # 将列表数据转成数据集形式
    dataset = tf.data.Dataset.from_tensor_slices(img_lab_files)

    # 将样本的顺序打乱
    if shuffle == True:
        dataset = dataset.shuffle(buffer_size=len(img_lab_files))

    # 对每个样本进行再加工
    # _parse_function()图像解码函数
    dataset = dataset.map(lambda x: _parse_function(x, config))

    # 根据设定的批次大小，对数据集进行填充
    dataset = dataset.padded_batch(config['batchsize'],
                                   padded_shapes=([config['tagsize'][0],
                                                   config['tagsize'][1], 1],
                                                  [config['label_len']]),
                                   drop_remainder=True)
    # 将样本的顺序打乱
    if repeat == True:
        dataset = dataset.repeat()

    # 设置数据集的缓存大小
    if isNewAPI == True:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.prefetch(1)

    print('___________________', len(img_lab_files), config['batchsize'])
    # 返回数据集和该数据集中所含的批次个数
    return dataset, len(img_lab_files) // config['batchsize']


# 定义函数将标签索引转成字符
def decodeindex(characters, indexs):
    result_str = ''.join(['' if c < 0 else characters[c] for c in indexs])
    return result_str


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # 2.0以下需要手动打开动态图
    assert LooseVersion(tf.__version__) >= LooseVersion("2.0")

    dataset_config = {
        'batchsize': 64,                       # 指定批次
        'image_path': r'dataimgcrop/images/',  # 指定图片文件路径
        'gt_path': r'dataimgcrop/gts/',        # 指定标注文件路径
        'tagsize': [31, 200],                  # 设置输出图片的高和宽
        'ch': 1,                               # 定义数据集输出的 图片通道数
        'label_len': 16,                       # 设置标签序列的总长度
        'characters': '0123456789' + string.ascii_letters,  # 用于将标签转化为索引的字符集（10个数字 + 26个小写字母 + 26个大写字母）
        'charoffset': 3,                       # 定义索引中的预留值：0 = 保留索引 、1 = pad 、2 = unk
    }

    # 获取图片数据集
    image_dataset, lengthdata = get_dataset(dataset_config, shuffle=False)

    for i, j in image_dataset.take(10):
        # 将归一化的数据转化成像素
        arraydata = np.uint8(np.squeeze(i[0].numpy() * 255))
        #        arraydata = np.uint8(  np.squeeze( (i[0].numpy()+1) * 127.5)  )

        lab_str = decodeindex(dataset_config['characters'], j[0].numpy().astype(int) - dataset_config['charoffset'])

        # 将标题和图片的内容显示出来
        plt.title('real label: %s\nlabel:%s' % (lab_str, j[0].numpy()))
        # plt.axis('off')

        plt.imshow(arraydata, cmap='gray')

        plt.show()
