from openslide import open_slide, __library_version__ as openslide_version
from skimage.color import rgb2gray
from skimage.io import imread
from scipy.misc import imsave
from scipy import misc
from numpy.random import random, choice, randint
import numpy as np
import os


def get_slide_tumor_mask(slide_path, tumor_mask_path):
    slide = open_slide(slide_path)
    tumor_mask = open_slide(tumor_mask_path)

    for i in range(min(len(tumor_mask.level_dimensions), len(slide.level_dimensions))):
        assert tumor_mask.level_dimensions[i][0] == slide.level_dimensions[i][0]
        assert tumor_mask.level_dimensions[i][1] == slide.level_dimensions[i][1]

    width, height = slide.level_dimensions[7]
    assert width * slide.level_downsamples[7] == slide.level_dimensions[0][0]
    assert height * slide.level_downsamples[7] == slide.level_dimensions[0][1]
  
    return slide, tumor_mask

def read_slide(slide, x, y, level, width, height, as_float=False):
    im = slide.read_region((x,y), level, (width, height))
    im = im.convert('RGB')
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im

def get_tissue_percentage(image, intensity=0.8):
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)
    return len(indices[0]) / float(image.shape[0] * image.shape[1]) * 100 

def get_patches(slide, tumor_mask, width, height, mask_width, mask_height, stride, n_level, path):
    path = '%s/level_%d' % (path, n_level)
    mask_x_offset, mask_y_offset = (width - mask_width) // 2, (height - mask_height) // 2
    level2xy_count = {}
    for level in range(8):
        level2xy_count[level] = {}
        level2xy_count[level]['x_count'] = (slide.level_dimensions[level][0] - width) // stride + 1
        level2xy_count[level]['y_count'] = (slide.level_dimensions[level][1] - height) // stride + 1  
  
    fp = open('%s/label.txt' % path, 'w')
    idx = 0
    for level, xy_count in level2xy_count.items():
        if level != n_level:
            continue
        factor = 2 ** level
        x_count, y_count = xy_count['x_count'], xy_count['y_count']
        for xi in range(x_count):
            for yi in range(y_count):
                x, y = xi * stride, yi * stride
                mask_x, mask_y = x + mask_x_offset, y + mask_y_offset
                slide_image = read_slide(slide, x * factor, y * factor, level, width, height)
                mask_image = read_slide(tumor_mask, mask_x * factor, mask_y * factor, level, mask_width, mask_height)[:,:,0]
        
                tissue_percentage = get_tissue_percentage(slide_image)
        
                if tissue_percentage < 20:
                    continue
        
                if np.sum(mask_image) > 0:
                    label = 'tumor'
                else:
                    label = 'normal'
                image_path = '%s/%s/%d.png' % (path, label, idx)
                imsave(image_path, slide_image)
                fp.write('%s\t%s\t%d\t%d\n' % (image_path, label, x, y))
                idx += 1
    fp.close()

def build_directory(root, level, label=True):
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists('%s/level_%d' % (root, level)):
        os.mkdir('%s/level_%d' % (root, level))
    if label:
        if not os.path.exists('%s/level_%d/normal' % (root, level)):
            os.mkdir('%s/level_%d/normal' % (root, level))
        if not os.path.exists('%s/level_%d/tumor' % (root, level)):
            os.mkdir('%s/level_%d/tumor' % (root, level))

def split_train_val(label_file, train_label_file, val_label_file):
    fp_train = open(train_label_file, 'w')
    fp_val = open(val_label_file, 'w')

    for line in open(label_file):
        if random() < 0.8:
            fp_train.write(line)
        else:
            fp_val.write(line)

    fp_train.close()
    fp_val.close()

def sample(train_label_file, sampled_train_label_file, n_samples):
    fp = open(sampled_train_label_file, 'w')
    normal_image, tumor_image = [], []
    for line in open(train_label_file):
        image_path, label, x, y = line.strip().split('\t')
        if label == 'normal':
            normal_image.append(line)
        elif label == 'tumor':
            tumor_image.append(line)
        else:
            print('error')
    for _ in range(n_samples):
        if random() < 0.5:
            line = choice(normal_image)
        else:
            line = choice(tumor_image)
        fp.write(line)
    fp.close()

def augment(image):
    i, j = randint(4), randint(2)
    while i > 0:
        image = np.rot90(image)
        i -= 1
    while j > 0:
        image = np.flip(image, axis=0)
        j -= 1
    return image

def image_generator(label_file, batch_size):
    label_2_binary = {'tumor': 1, 'normal': 0}
    filenames, labels = [], []
    for line in open(label_file):
        image_path, label = line.strip().split('\t')[:2]
        filenames.append(image_path)
        labels.append(label_2_binary[label])
    n = len(filenames)
    while True:
        selected_index = np.random.choice(n, size=batch_size)
        
        batch_input = []
        batch_output = []
        
        for idx in selected_index:
            image_path = filenames[idx]
            image = imread(image_path) / 255.0
            
            image = augment(image)
            label = labels[idx]
            
            batch_input.append(image)
            batch_output.append(label)
        
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        
        yield(batch_x, batch_y)
   