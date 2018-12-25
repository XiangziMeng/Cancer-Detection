import requests
import os

def download_data(slide_path, tumor_mask_path, slide_url, mask_url):
  # Download the whole slide image
  if not os.path.exists(slide_path):
    r = requests.get(slide_url, allow_redirects=True)
    open(slide_path, 'wb').write(r.content)
    
  # Download the tumor mask
  if not os.path.exists(tumor_mask_path):
    r = requests.get(mask_url, allow_redirects=True)
    open(tumor_mask_path, 'wb').write(r.content)
    
if __name__ == '__main__':
    data_folder = './data'
    slide_path = 'tumor_091.tif'
    tumor_mask_path = 'tumor_091_mask.tif'
    
    slide_url = 'https://storage.googleapis.com/applied-dl/%s' % slide_path
    mask_url = 'https://storage.googleapis.com/applied-dl/%s' % tumor_mask_path
    
    test_slide_path = 'tumor_101.tif'
    test_tumor_mask_path = 'tumor_101_mask.tif'

    test_slide_url = 'https://storage.googleapis.com/project_xiangzi_adl/%s' % test_slide_path
    test_mask_url = 'https://storage.googleapis.com/project_xiangzi_adl/%s' % test_tumor_mask_path

        
    # Download the training and testing slides
    download_data('%s/%s' % (data_folder, slide_path), '%s/%s' % (data_folder, tumor_mask_path), slide_url, mask_url)
    download_data('%s/%s' % (data_folder, test_slide_path), '%s/%s' % (data_folder, test_tumor_mask_path), test_slide_url, test_mask_url)