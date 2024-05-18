import my_cv
import time
import cv2
from SA import sa
import numpy as np
import atexit
import subprocess

def run_next_program():
    subprocess.run(['python', "路径\\Road.py"])#第二个函数是下一个程序（路径规划）的位置


def undistort(Frame):
    fx = 469.661045287806
    cx = 314.9348873078631
    fy = 470.04020805530286
    cy = 262.6958372196564
    k1, k2, p1, p2, k3 = -0.3970199011336049, 0.15556505640757545, -0.0010010243380322113, 0.00019479138742734106, -0.031021065963771827

    # 相机坐标系到像素坐标系的转换矩阵
    k = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    # 畸变系数
    d = np.array([
        k1, k2, p1, p2, k3
    ])
    h, w = Frame.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), 5)
    return cv2.remap(Frame, mapx, mapy, cv2.INTER_LINEAR)


if __name__ == '__main__':
    sta_x = 20
    sta_y = 1
    tar_x = 0
    tar_y = 19

    map_i = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
             [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
             [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
             [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
             [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
             [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
             [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
             [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
             [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
             [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
             [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
             [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
             [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
             [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
             [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    cap1 = cv2.VideoCapture(0)

    width = 640
    height = 480

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # ser = serial.Serial("/dev/ttyS0", 9600, timeout=0.1)

    real_point_list = []
    cmd = 0
    print("track_cap1 is open ？ {}".format(cap1.isOpened()))

    while True:
        # 成功识别后按下回车   
        
        
        ret, _img = cap1.read()
        # _img = undistort(_img)
        # 如果正确读取帧，ret为True
        if not ret:
            print("Can't receive frame (stream)")
            continue
        # _img = cv2.imread('book1.png')
        # 得到四个方框点和四个地图点
        raw_point = my_cv.get_four_cor(_img)
        if raw_point == 0:
            print("Can't receive picture")
            cmd = cv2.waitKey(5)
            continue
        # 将四个方框点按 左上、右上、左下、右下 排序
        right_point = my_cv.range_four_col(raw_point)
        # 根据方框点计算透视矩阵
        _M, _M_inverse = my_cv.cal_perspective_params(_img, right_point)
        # 透视转换
        tras_img = my_cv.img_perspect_transform(_img, _M)
        # cv2.imshow('tras_img', tras_img)
        # cv2.waitKey(0)
        # 根据变换后的图像寻找最大轮廓
        inner_img = my_cv.get_inner_col(tras_img)

        # cv2.imshow('inner_img', inner_img)
        # cv2.waitKey(0)
        # 寻找霍夫圆
        get_point_list = my_cv.get_circle_cor(inner_img)
        if len(get_point_list) != 8:
            print("cannot find all circle")
            cmd = cv2.waitKey(5)
            continue
        for point in get_point_list:
            if map_i[point[1]][point[0]] == 1:
                print("find wrong:")
                print(point)
                continue
        real_point_list = get_point_list
        cmd = cv2.waitKey(5)
        if True:
            T1 = time.time()
            all_Path, all_Ex, best_Way = sa(10, pow(10, -1), 0.9, 50, len(real_point_list), real_point_list, map_i,
                                            sta_x,
                                            sta_y,
                                            tar_x, tar_y)
            T2 = time.time()
            print('程序运行时间:%s毫秒' % ((T2 - T1) * 1000))
            print(best_Way)
            print(len(best_Way))
            ts = [[1,9],[3,19],[5,15],[5,9],[5,7],[7,19],[7,13],[7,11],[7,9],[7,5],[9,3],[9,17],[11,17],[13,15],[13,11],[13,9],[13,7],[13,1],[15,5],[15,11],[15,13],[17,1],[19,11]]
        
            list = []
            last = 'L'
            sum = 0
            x = 0
            y = 1
            for i in best_Way:
                print(x,y,i)
                # if last==i:
                    # continue
                # e  
                if [x,y] in ts:
                    if last==i:
                        list.append(1)
                if last != i:
                    sum+=1
                    if last == 'L' and i == 'D' or last=='R' and i=='U' or last=='U' and i=='L' or last=='D' and i=='R':
                        # last=i
                        list.append(0)
                    elif last =='L' and i=='U' or last=='R' and i=='D' or last=='U' and i=='R' or last=='D' and i=='L':
                        # last=i
                        list.append(2)                   
                    elif i == '#':
                         list.append(3)
                        #  last=i
                    last=i
                if i == 'L':
                    x+=1
                elif i == 'R':
                    x-=1
                elif i == 'U':
                    y-=1
                elif i == 'D':
                    y+=1
            # ser.write(best_Way)
            print(sum)
            print(list)
            print(len(list))
            data = str(list)
            file = open('路径\\road.txt','w')
            file.write(data)
            print("数据已写入")
            file.close()
            atexit.register(run_next_program)
            break
