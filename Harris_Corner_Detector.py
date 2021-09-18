from scipy import signal
from numpy.linalg import det
import numpy as np
import time


def HarrisCornerDetection(image):
    Harris_time_start = time.time()
    w, h = image.shape
    ImgX, ImgY = sobel_edge(image)

    # # Eliminate the negative values why ??
    for ind1 in range(w):
        for ind2 in range(h):
            if ImgY[ind1][ind2] < 0:
                ImgY[ind1][ind2] *= -1
            if ImgX[ind1][ind2] < 0:
                ImgX[ind1][ind2] *= -1

   # calculate the element of M matrix
    ImgX_2 = np.square(ImgX)
    ImgY_2 = np.square(ImgY)
    ImgXY = np.multiply(ImgX, ImgY)
    ImgYX = np.multiply(ImgY, ImgX)

    #Use Gaussian Blur
    Sigma = 1.4
    kernelsize = (3, 3)

    ImgX_2 = gaussian_filter(ImgX_2, kernelsize, Sigma)
    ImgY_2 = gaussian_filter(ImgY_2, kernelsize, Sigma)
    ImgXY = gaussian_filter(ImgXY, kernelsize, Sigma)
    ImgYX = gaussian_filter(ImgYX, kernelsize, Sigma)

    alpha = 0.06
    R = np.zeros((w, h), np.float32)
    # For every pixel find the corner strength
    for row in range(w):
        for col in range(h):
            M_bar = np.array([[ImgX_2[row][col], ImgXY[row][col]], [ImgYX[row][col], ImgY_2[row][col]]])
            R[row][col] = np.linalg.det(M_bar) - (alpha * np.square(np.trace(M_bar)))

    # Empirical Parameter
    # This parameter will need tuning based on the use-case
    CornerStrengthThreshold = 600000

    Key_Points = []
    # Look for Corner strengths above the threshold
    for row in range(w):
        for col in range(h):
            if R[row][col] > CornerStrengthThreshold:
                # print(R[row][col])
                max = R[row][col]

                # Local non-maxima suppression
                skip = False
                for nrow in range(5):
                    for ncol in range(5):
                        if row + nrow - 2 < w and col + ncol - 2 < h:
                            if R[row + nrow - 2][col + ncol - 2] > max:
                                skip = True
                                break

                if not skip:
                    # Point is expressed in x, y which is col, row
                    # cv2.circle(bgr, (col, row), 1, (0, 0, 255), 1)
                    Key_Points.append((row, col))
    Harris_time_end = time.time()
    print(f"Execution time of the Harris corner Detector is {Harris_time_end - Harris_time_start}  sec")
    return Key_Points

# helper
def gaussian_filter(img, shape, sigma):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    return signal.convolve2d(img, h)

def sobel_edge(img):
    row, col = img.shape
    Ix = np.zeros([row, col])
    Iy = np.zeros([row, col])

    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    ky = (kx.transpose())
    for i in range(1, row - 2):
        for j in range(1, col - 2):
            Ix[i][j] = np.sum(np.multiply(kx, img[i:i + 3, j:j + 3]))
            Iy[i][j] = np.sum(np.multiply(ky, img[i:i + 3, j:j + 3]))

    return Ix, Iy

"""""
#### test ####

# Get the first image
firstimage = cv2.imread('images/Cow.PNG', cv2.IMREAD_GRAYSCALE)
w, h = firstimage.shape
# Covert image to color to draw colored circles on it
bgr = cv2.cvtColor(firstimage, cv2.COLOR_GRAY2RGB)
# Corner detection
R = HarrisCornerDetection(firstimage)

# Display image indicating corners and save it
cv2.imshow("Corners", bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""""