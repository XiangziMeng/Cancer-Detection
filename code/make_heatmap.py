from utils import *
import matplotlib.pyplot as plt
data_folder = './data'

test_slide_path = '%s/tumor_101.tif' % data_folder
test_tumor_mask_path = '%s/tumor_101_mask.tif' % data_folder

test_slide, test_tumor_mask = get_slide_tumor_mask(test_slide_path, test_tumor_mask_path)

test_slide_level_7 = read_slide(test_slide, 0, 0, 7, test_slide.level_dimensions[7][0], test_slide.level_dimensions[7][1])
test_tumor_mask_level_7 = read_slide(test_tumor_mask, 0, 0, 7, test_tumor_mask.level_dimensions[7][0], test_tumor_mask.level_dimensions[7][1])[:, :, 0]


plt.imshow(test_tumor_mask_level_7)
plt.imshow(test_slide_level_7, cmap='jet', alpha=0.5) 
plt.savefig('./heatmap/ground_truth_heatmap.png', dpi=300)


heat_map = np.ones((560, 1088, 3))
threshold = 0.35
for line in open('./prediction/test_prediction.txt'):
    image_path, label, x, y, score = line.strip().split('\t')
    x, y, score = int(y) // 8, int(x) // 8, float(score)
    if score > threshold:
        for x_pixel in range(x, x + 8):
            for y_pixel in range(y, y + 8):
                heat_map[x_pixel][y_pixel][1] = 255
                
                
plt.imshow(heat_map[:, :, 1])
plt.imshow(test_slide_level_7, cmap='jet', alpha=0.5) 
plt.savefig('./heatmap/predicted_heatmap', dpi=300)                

print('Finish making heatmaps')