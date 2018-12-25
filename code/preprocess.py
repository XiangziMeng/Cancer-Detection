from utils import *

def data_preprocessing(data_folder, slide_path, tumor_mask_path, test_slide_path, test_tumor_mask_path, \
                      width, height, mask_width, mask_height, stride, n_level):
    """Extract patches from slides, make splits of training, validating and testing data, sample data from training data"""
    slide, tumor_mask = get_slide_tumor_mask(slide_path, tumor_mask_path)
    test_slide, test_tumor_mask = get_slide_tumor_mask(test_slide_path, test_tumor_mask_path)
    
    print('build directories')
    
    build_directory(root='%s/all_data' % data_folder, level=n_level, label=True)
    build_directory(root='%s/test_data' % data_folder, level=n_level, label=True)
    build_directory(root='%s/train' % data_folder, level=n_level, label=False)
    build_directory(root='%s/val' % data_folder, level=n_level, label=False)
    build_directory(root='%s/sampled_train' % data_folder, level=n_level, label=False)   

    label_file = '%s/all_data/level_%d/label.txt' % (data_folder, n_level)
    train_label_file = '%s/train/level_%d/label.txt' % (data_folder, n_level)
    val_label_file = '%s/val/level_%d/label.txt' % (data_folder, n_level)
    sampled_train_label_file = '%s/sampled_train/level_%d/label.txt' % (data_folder, n_level)
    
    print('make patches')
    
    get_patches(slide, tumor_mask, width, height, mask_width, mask_height, stride, \
                n_level, '%s/all_data' % data_folder)
    get_patches(test_slide, test_tumor_mask, width, height, mask_width, mask_height, stride, \
                n_level, '%s/test_data' % data_folder)
    
    print('split training and validating images')
    
    split_train_val(label_file, train_label_file, val_label_file)
    
    cnt = 0
    for line in open(train_label_file):
        cnt += 1
    n_samples = (cnt // 100 + 1) * 100
    
    print('data sampling')
    
    sample(train_label_file, sampled_train_label_file, n_samples)

    print('finish preprocessing')
    
if __name__ == '__main__':
    data_folder = './data'
    slide_path = '%s/tumor_091.tif' % data_folder
    tumor_mask_path = '%s/tumor_091_mask.tif' % data_folder
    test_slide_path = '%s/tumor_101.tif' % data_folder
    test_tumor_mask_path = '%s/tumor_101_mask.tif' % data_folder
    
    # Prepare training, validating and testing data on Level 4
    data_preprocessing(data_folder, slide_path, tumor_mask_path, test_slide_path, test_tumor_mask_path, \
                      width=100, height=100, mask_width=64, mask_height=64, stride=64, n_level=4)
