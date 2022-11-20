import cv2
from pathlib import Path
import re
import os
import numpy as np

tolerance = 20

SUCCESSFUL = '\033[92m'
END = '\033[0m'

# the goal size and image type are defined here
size = np.array([228, 228])
img_type = 'jpg'

# all the image types that occur in our data
file_types = ['jpg', 'png', 'jfif']


# https://stackoverflow.com/questions/53732747/set-white-background-for-a-png-instead-of-transparency-with-opencv/53737420
def remove_transparency(img):
    # Make mask of transparent pixels
    mask = img[:,:,3] == 0
    img[mask] = [255, 255, 255, 255]

    return img


def fill_background(img):
    height, width, c = img.shape

    source_color = (255, 255, 255)

    to_fill = set()
    to_fill.add((0, 0))
    while len(to_fill) != 0:
        (x, y) = to_fill.pop()

        if not is_pixel_in_bounds(width, height, x, y):
            continue

        diff = get_color_difference(img[y, x], source_color)

        if diff == 0 or diff > tolerance:
            continue

        img[y, x] = source_color

        to_fill.add((x - 1, y))
        to_fill.add((x + 1, y))
        to_fill.add((x, y - 1))
        to_fill.add((x, y + 1))

    return img


def is_pixel_in_bounds(width, height, x, y):
    return 0 <= x < width and 0 <= y < height


# Reference:
# https://stackoverflow.com/questions/9018016/how-to-compare-two-colors-for-similarity-difference
def get_color_difference(c1, c2):

    blue_diff = abs(int(c1[0]) - int(c2[0])) / 255
    green_diff = abs(int(c1[1]) - int(c2[1])) / 255
    red_diff = abs(int(c1[2]) - int(c2[2])) / 255

    difference = (blue_diff + green_diff + red_diff) / 3 * 100
    return difference


def clean_dir(path_to_dir):
    for f_type in file_types:
        for img in path_to_dir.glob(f'**\*.{f_type}'):
            # load image
            img_path = str(img)
            np_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if np_img.shape[0] == size[0] and np_img.shape[1] == size[1]:
                continue

            # if image has 4 channels, remove transparency
            if len(np_img[0][0]) == 4:
                np_img = remove_transparency(np_img)
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

            # file type
            pattern = '(.*\.)(jpg|JPG|png|jfif)'
            cleaned_img_path = re.match(pattern, img_path).group(1) + img_type

            # resize
            resize_factors = size / np_img.shape[0:2]

            smaller_resize_factor_dim = resize_factors.tolist().index(min(resize_factors[0:2]))
            larger_resize_factor_dim = (smaller_resize_factor_dim - 1) % 2

            np_img = cv2.resize(np_img, (0, 0), fx=resize_factors[smaller_resize_factor_dim], fy=resize_factors[smaller_resize_factor_dim])

            # fill background (embedded in resize part)
            np_img = fill_background(np_img)

            empty_image = np.full((size[0], size[1], 3), 255, np.uint8)

            offset = (size[larger_resize_factor_dim] - np_img.shape[larger_resize_factor_dim]) // 2
            for x in range(np_img.shape[0]):
                for y in range(np_img.shape[1]):
                    empty_image[x + offset*smaller_resize_factor_dim][y + offset*larger_resize_factor_dim] = np_img[x][y]
            np_img = empty_image

            # remove old image and write new image
            os.remove(img_path)
            cv2.imwrite(cleaned_img_path, np_img)
            print(f'{SUCCESSFUL}{img_path}      cleaned successfully!{END}')


def run():
    path_to_main_dir = Path('./dataset')
    clean_dir(path_to_main_dir)


if __name__ == "__main__":
    run()