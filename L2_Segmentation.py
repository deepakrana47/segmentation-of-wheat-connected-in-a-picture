import cv2, numpy as np
from Area import areaThreshold_by_havg, areaThreshold_by_avg
from threshold import otsu_threshold
from _8connected import get_8connected_v2
from util import *
import warnings
import traceback
warnings.filterwarnings("error")

color = {i: np.random.randint(20, 255, 3) for i in range(5, 5000)}
color[1] = [255, 255, 255]
color[2] = [0, 0, 255]


def segment_image4(img_file):
    org = cv2.imread(img_file, cv2.IMREAD_COLOR)
    h, w = org.shape[:2]

    img=org.copy()

    # removing noise by using Non-local Means Denoising algorithm
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    # cv2.imshow('cleaned',img)

    # Taking the red component out of RBG image as it is less effected by shadow of grain or impurity
    gray = img[:,:,2]
    # cv2.imshow('gray',gray)

    # calculating threshold value by using otsu thresholding
    T = otsu_threshold(gray=gray)

    # incresing contrast about the threshold
    gray = np.array([[max(pixel - 25, 0) if pixel < T else min(pixel + 25, 255) for pixel in row] for row in gray], dtype=np.uint8)
    # cv2.imshow('contrast',gray)

    # generating a threshold image
    thresh = np.array([[0 if pixel<T else 255 for pixel in row]for row in gray], dtype=np.uint8)
    # cv2.imshow('Threshold',thresh)

    ########################## 1st level of segmentation ########################################
    # print " Level 1 segmentation"

    # generating a mask using 8-connected component method on threshold image
    mask = get_8connected_v2(thresh, mcount=5)
    # display_mask("Initial mask",mask)

    # Calcutaing the grain segment using mask image
    s = cal_segment_area(mask)

    # cv2.waitKey()
    # removing the backgraound of grain
    # timg = np.array([[[0,0,0] if mask[i,j] == 0 else org[i,j] for j in range(w)] for i in range(h)], dtype=np.uint8)

    # removing very small particals (smaller the 2^3 the average size)
    print(s)
    low_Tarea, up_Tarea = areaThreshold_by_havg(s, 3)
    slist = list(s)
    for i in slist:
        area = (s[i][0] - s[i][1]) * (s[i][2] - s[i][3])
        if area < low_Tarea or area > up_Tarea:
            s.pop(i)

    # print " Level 1 segmentation Finished"
    ####################### 1st level of segmentation Finished ##################################

    ####################### 2nd level of segmentation ###########################################
    # print "\t Level 2 segmentation"
    # print s
    new_s = {}
    s_range = [i for i in s]
    max_index = max(s_range)
    for sindex in s_range:
        if s[sindex][0] - s[sindex][1] and s[sindex][2] - s[sindex][3]:
            iimg = np.array([[img[i, j] if mask[i, j] == sindex else [0, 0, 0] for j in range(s[sindex][2], s[sindex][3])] for i in range(s[sindex][0], s[sindex][1])], dtype=np.uint8)
            if len(iimg) == 0:
                continue
            mask1 = L2_segmentation_2(iimg, T=T, index=max_index + 5 + len(new_s))
            if mask1 is None:
                continue
            ######################################## segmenting adding ########################
            m = s.pop(sindex)
            mask[m[0]:m[1], m[2]:m[3]] = [[0 if pixel == sindex else pixel for pixel in row] for row in mask[m[0]:m[1], m[2]:m[3]]]
            mask[m[0]:m[1], m[2]:m[3]] += mask1

            s1 = cal_segment_area(mask1)
            # print s1
            for k in s1:
                area = (s1[k][0] - s1[k][1]) * (s1[k][2] - s1[k][3])
                if area > low_Tarea and area < up_Tarea:
                    new_s[k] = [m[0] + s1[k][0], m[0] + s1[k][1], m[2] + s1[k][2], m[2] + s1[k][3]]
            max_index = max([max_index]+list(new_s))
            ###################################################################################

    # print "2nd Level of segmentation Finished"
    #####################2nd level of segmentation Finished ###################################
    s.update(new_s)

    # marking the segments
    segments = {}
    torg = org.copy()
    for i in s:
        imgRectangled = cv2.rectangle(torg, (s[i][2], s[i][0]), (s[i][3], s[i][1]), (0, 0, 255), 1)
        segments[i] = np.array([[org[x,y] if mask[x,y] == i else [0,0,0] for y in range(s[i][2],s[i][3])] for x in range(s[i][0],s[i][1])], dtype=np.uint8)
        # # cv2.imshow("segment %d" % (count), segments[count])

    # # cv2.imshow('Marked image',imgRectangled)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return segments, s, imgRectangled, mask

def L2_segmentation_2(iimg , T, index):
    h, w, _ = iimg.shape
    # cv2.imshow('image', iimg)


    gray = iimg[:, :, 2]
    # cv2.imshow('gray', gray)

    thresh = np.array([[0 if pixel < T else 255 for pixel in row] for row in gray], dtype=np.uint8)

    sober = sober_operation(gray)
    # cv2.imshow('sober', sober)

    sober = cv2.fastNlMeansDenoising(sober, None, h=2, templateWindowSize=3, searchWindowSize=5)
    # cv2.imshow('sober cleaned', sober)

    T= otsu_threshold(sober)

    sthresh = np.array([[0 if pixel < T else 255 for pixel in row] for row in sober], dtype=np.uint8)
    # cv2.imshow('sober Threshold', sthresh)

    diluted = cv2.dilate(sthresh, kernel=np.ones((5,5), np.uint8), iterations=1)
    # cv2.imshow('dilutated2 ', diluted)

    thresh2 = np.where((thresh == 0) * (diluted == 255), 0, thresh-diluted)
    # cv2.imshow('Thresh - dilute ', thresh2)

    mask = get_8connected_v2(thresh=thresh2, mcount=index)
    # display_mask("Diluted mask", mask)

    # Calcutaing the grain segment using mask image
    s = cal_segment_area(mask)

    if len(s) < 2:
        return None
    low_Tarea, up_Tarea = areaThreshold_by_avg(s, 2)
    slist = list(s)
    for i in slist:
        area = (s[i][0] - s[i][1]) * (s[i][2] - s[i][3])
        if area < low_Tarea or area > up_Tarea:
            s.pop(i)
    if len(s) < 2:
        return None

    # # # cv2.imshow("watershed", water_shed)

    # removing unwanted masks
    mask = np.array([[0 if pixel not in s else pixel for pixel in row] for row in mask])

    # Adding boundry mask
    boundry = get_boundry_img_matrix(thresh, 1)
    mask = np.where(boundry == 1, 1, mask)

    # display_mask('boundried mask', mask)

    # using mask fill the mask values in boundry
    mask = flood_filling(mask)
    # display_mask('flood fill', mask)

    # replace boundry by respective mask value
    mask = boundry_fill(mask)

    # display_mask("Final Mask",mask)
    # cv2.waitKey()

    # cv2.destroyAllWindows()
    return mask