import numpy as np, cv2


color = {i: np.random.randint(20, 255, 3) for i in range(3, 100000)}
color[1] = [255, 255, 255]
color[2] = [0, 0, 255]


def get_boundry_as_points(thresh):
    line = []
    h, w = thresh.shape
    border = np.zeros((h, w), dtype=np.uint8)
    for i in range(w):
        if thresh[0, i] > 0:
            border[0, i] = thresh[0, i]
        if thresh[h - 1, i] > 0:
            border[h - 1, i] = thresh[h - 1, i]
    for i in range(1, h):
        if thresh[i, 0] > 0:
            border[i, 0] = thresh[i, 0]
        if thresh[i, w - 1] > 0:
            border[i, w - 1] = thresh[i, w - 1]
        for j in range(1, w):
            if int(thresh[i, j]) - thresh[i, j - 1] > 0 or int(thresh[i, j]) - thresh[i - 1, j] > 0:
                border[i, j] = thresh[i, j]
            elif int(thresh[i, j]) - thresh[i, j - 1] < 0:
                border[i, j - 1] = thresh[i, j - 1]
            if int(thresh[i, j]) - thresh[i - 1, j] < 0:
                border[i - 1, j] = thresh[i - 1, j]
    for i in range(h):
        if border[i, int(w / 2)] != 0: break
    start = (i, int(w / 2))
    lstart = (i, int(w / 2))
    line.append(list(lstart))
    sign = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    ind = 1
    slen = len(sign)
    while 1:
        count = 0
        while 1:
            if lstart[0] + sign[ind][0] < 0 or lstart[1] + sign[ind][1] < 0 or lstart[0] + sign[ind][0] > h - 1 or \
                                    lstart[1] + sign[ind][1] > w - 1:
                ind = (ind + 1) % slen;
                count += 1
                continue
            if (border[lstart[0] + sign[ind][0], lstart[1] + sign[ind][1]] == 0 and count < slen - 1):
                ind = (ind + 1) % slen;
                count += 1
            else:
                break
        if (lstart[0] + sign[ind][0], lstart[1] + sign[ind][1]) == start:
            break
        lstart = (lstart[0] + sign[ind][0], lstart[1] + sign[ind][1])
        ind = (sign.index((-1 * sign[ind][0], -1 * sign[ind][1])) + 1) % slen
        line.append(list(lstart))
    return line

def get_boundry_img_matrix(thresh, bval=1):
    h, w = thresh.shape
    thresh = padding2D_zero(thresh,1)
    border = np.zeros(thresh.shape, dtype=np.uint8)
    for i in range(1,h+1):
        for j in range(1,w+1):
            if thresh[i,j] == 0 and border[i,j] != bval:
                if thresh[i,j+1] > 0:
                    border[i,j+1] = 1
                if thresh[i+1,j] > 0:
                    border[i+1,j] = 1
                if thresh[i,j-1] > 0:
                    border[i, j-1] = 1
                if thresh[i - 1, j] > 0:
                    border[i-1, j] = 1
    for i in range(1,h+1):
        if thresh[i,1] > 0: border[i,1] = 1
        if thresh[i,w] > 0: border[i,w] = 1
    for j in range(1,w+1):
        if thresh[1,j] > 0: border[1,j] = 1
        if thresh[h,j] > 0: border[h,j] = 1

    # thresh = remove_padding2D_zero(thresh,1)
    border = remove_padding2D_zero(border,1)*bval
    return border

def padding2D_zero(matrix, num=1, dtype=np.float32):
    h, w = matrix.shape
    matrix2 = np.concatenate((np.zeros((num, w), dtype=dtype), matrix, np.zeros((num, w), dtype=dtype)), axis=0)
    matrix2 = np.concatenate((np.zeros((h + 2*num, num)), matrix2, np.zeros((h + 2*num, num))), axis=1)
    return matrix2


def remove_padding2D_zero(matrix, num):
    return matrix[num:-num, num:-num]


def generate_newcolorimg_by_padding(img, newh, neww):
    h,w = img.shape[0:2]
    # print "original size:", img.shape[0:2],
    if h > newh or w > neww:
        if h > newh and w > neww:
            if newh*w/h > neww:
                dim = (int(neww*h/w),neww)
            else:
                dim = (newh, int(newh*w/h))
        elif h > newh:
            dim = (int(newh * w / h),newh)
        else:
            dim = (neww, int(neww * h / w))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

    h,w,c = img.shape
    # print "after resize:",img.shape[0:2],
    h0 = newh - h
    w0 = neww - w
    h1 = int(h0/2)
    w1 = int(w0/2)
    newimg = img.copy()
    if w0 != 0:
        left_pad = np.zeros((h, w1, c), dtype=np.uint8)
        right_pad = np.zeros((h, w0-w1, c), dtype=np.uint8)
        newimg = np.concatenate((left_pad, img, right_pad), axis=1)
    if h0 != 0:
        top_pad = np.zeros((h1, neww, c), dtype=np.uint8)
        bottom_pad = np.zeros((h0 - h1, neww, 3), dtype=np.uint8)
        newimg = np.concatenate((top_pad, newimg, bottom_pad), axis=0)
    # print "new size:",newimg.shape[0:2]
    return newimg

def sober_operation(img):
    h, w = img.shape
    d = np.array([[1, 3, 1], [0, 0, 0], [-1, -3, -1]])
    val=1
    # d = np.array([[1, 8, 10, 8, 5], [4,10,20,10,4], [0, 0, 0,0,0], [-1, -8, -10, -8, -5], [-4,-10,-20,-10,-4]])
    imgn = padding2D_zero(img, val)
    gx = np.zeros(imgn.shape)
    gy = np.zeros(imgn.shape)
    # try:
    for i in range(val, h+val):
        for j in range(val, w+val):
            gx[i, j] = np.sum(np.multiply(imgn[i - val:i + val+1, j - val:j + val+1], d))
            gy[i, j] = np.sum(np.multiply(imgn[i - val:i + val+1, j - val:j + val+1], d.T))
    # except ValueError:
    #     print
    gx = remove_padding2D_zero(gx, val)
    gy = remove_padding2D_zero(gy, val)
    grad = np.sqrt(np.square(gx) + np.square(gy))
    return grad.astype(np.uint8)

def cal_segment_area(mask):
    h,w = mask.shape
    s={}
    for i in range(h):
        for j in range(w):
            if mask[i,j]:
                if mask[i,j] in s:
                    if i < s[mask[i,j]][0]: s[mask[i, j]][0] = i
                    elif i > s[mask[i,j]][1]: s[mask[i, j]][1] = i
                    if j < s[mask[i,j]][2]: s[mask[i, j]][2] = j
                    elif j > s[mask[i,j]][3]: s[mask[i, j]][3] = j
                if mask[i,j] not in s:
                    s[mask[i,j]] = [i,i,j,j]

    for m in s:
        s[m][1] += 1
        s[m][3] += 1
    return s

def flood_filling(mask1):
    ####################################### filling color ##########################
    # print "\tfilling Colour"
    h, w = mask1.shape
    ival=val = 3
    mask1 = padding2D_zero(mask1, ival)
    pcount = 0
    ite=0
    while 1:
        count = 0
        temp = mask1.copy()
        for i in range(val, h + val):
            for j in range(val, w + val):
                if mask1[i,j] > 2:
                    if not np.any(mask1[i,j+1:j+val+1]):
                        temp[i,j+1:j+val+1] = np.ones(val)*mask1[i,j]

                    if not np.any(mask1[i,j-val:j]):
                        temp[i,j-val:j] = np.ones(val)*mask1[i,j]

                    if not np.any(mask1[i+1:i+val+1,j]):
                        temp[i+1:i+val+1,j] = np.ones(val)*mask1[i,j]

                    if not np.any(mask1[i-val:i,j]):
                        temp[i-val:i,j] = np.ones(val)*mask1[i,j]
                elif mask1[i,j] == 0:
                    count += 1
        if count == pcount:
            if val == 1:
                break
            val -= 1
        # print val, count
        pcount = count
        # display_mask('inside mask',temp)
        # cv2.waitKey()
        mask1 = temp.copy()
        ite+=1
    mask1 = remove_padding2D_zero(mask1, ival)
    return mask1

def get_mask_value_area(img, mask, mval):
    h,w = img.shape
    iimg = np.zeros(img.shape, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if mask[i,j] == mval:
                iimg[i,j] = img[i,j]
    return iimg

def display_mask(name, mask, sname=None):
    mask_section = formMarkimg(mask)
    cv2.imshow(name, mask_section)
    # cv2.waitKey()
    if sname:
        cv2.imwrite(sname, mask_section)
    return

def formMarkimg(mask):
    return np.array([[color[pixel] if pixel else [0, 0, 0] for pixel in row] for row in mask], dtype = np.uint8)

def invert_gray(img):
    return np.array([[255-pixel for pixel in row] for row in img], dtype=np.uint8)

def boundry_fill(mask):
    h,w = mask.shape
    mask= padding2D_zero(mask, 1)
    bound = 1
    ite = 15
    while bound and ite:
        bound = 0
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                if mask[i,j] == 1:
                    bound+=1
                    if mask[i,j+1] > 4:
                        mask[i,j] = mask[i,j+1]
                    elif mask[i+1,j] > 4:
                        mask[i,j] = mask[i+1,j]
                    elif mask[i,j-1] > 4:
                        mask[i,j] = mask[i,j-1]
                    elif mask[i-1,j] > 4:
                        mask[i,j] = mask[i-1,j]
        ite-=1
    mask = remove_padding2D_zero(mask, 1)
    return mask

def otsu_threshold(gray):
    h, w = gray.shape
    count = {i: 0 for i in range(256)}
    for i in range(h):
        for j in range(w):
            count[gray[i, j]] += 1
    prob = np.array([count[i] / float(h * w) for i in sorted(count)])
    means = np.array([prob[i] * (i + 1) for i in count])
    mean = np.sum(means)
    minvar = -np.inf
    minT = 0
    for t in range(256):
        w1 = np.sum([i for i in prob[:t + 1]])
        w2 = 1.0 - w1
        if not w1 or not w2: continue
        m1 = np.sum([i for i in means[:t + 1]])
        mean1 = m1 / w1
        mean2 = (mean - m1) / w2
        bcvar = w1 * w2 * (mean2 - mean1) ** 2
        if bcvar > minvar:
            minvar = bcvar
            minT = t
    return minT