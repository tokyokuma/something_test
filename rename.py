import os
import glob

dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/qhd_color'
filelist = glob.glob('/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/qhd_color/*.png')
i = 17817

for file in filelist:
    print file
    os.rename(file, os.path.join(dir,'frame%06.f.png'%i))
    i = i + 1
