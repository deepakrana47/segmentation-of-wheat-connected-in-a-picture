import os
from util import *
from L2_Segmentation_v3 import segment_image4

def get_files(indir):
    indir = indir.rstrip('/')
    flist =os.listdir(indir)
    files = []
    for f in flist:
        f = indir+'/'+f
        if os.path.isdir(f):
            tfiles = get_files(f)
            files += [tf for tf in tfiles]
        else:
            files.append(f)
    return files

if __name__ == "__main__":
    in_dir = 'over_lapping_grain/'
    img_files = get_files(in_dir)
    out = 'Result_grain/'
    if not os.path.isdir(out): os.mkdir(out)
    count = 0
    for img in img_files:
        print img
        seg, s, imgRectange, mask= segment_image4(img)
        print s
        if not seg: continue
        o = out+img.split('.')[0][len(in_dir):]+'_'
        for i in s:
            cv2.imwrite(o+str(i)+'.jpg', seg[i])
            # # cv2.imshow('segment_%d'%(i),seg[i])
        cv2.imwrite(o+'mask.jpg', formMarkimg(mask))

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
