import itertools
import matplotlib.pyplot as plt
import time
# import serial

T1 = time.time()


class Node(object): #定义一个节点  class 类似c语言里面的结构体
    def __init__(self, x, y, w): #初始化对象的值
        self.x = x
        self.y = y
        self.w = w

    def __str__(self): #定义一个函数 str是返回字符型
        return self.w


def up(node): #定义向上函数
    return Node(node.x - 1, node.y, node.w + "L")


def down(node):#定义向下下函数
    return Node(node.x + 1, node.y, node.w + "R")


def left(node):#定义往左函数
    return Node(node.x, node.y - 1, node.w + "U")


def right(node):#定义向右函数
    return Node(node.x, node.y + 1, node.w + "D")


def get_way(start_x, start_y, target_x, target_y, map_int): #初始化路
    m = len(map_int[0]) #表示迷宫第一行的长度
    n = len(map_int)#表示迷宫一共有几行 类似于迷宫的高

    queue = []
    visited = []

    node = Node(start_x, start_y, "") #设置起始点为起点 w为空
    queue.append(node) #把节点添加到队列里面
    while len(queue) != 0: #如果节点全处理完则跳出循环
        moveNode = queue[0] #从队列里取出第一个节点
        queue.pop(0)#移除队列里第一个节点
        moveStr = str(moveNode.x) + " " + str(moveNode.y) #创建一个字符串表示当前坐标 中间的空格便于区分
        if moveStr not in visited: #如果当前节点没有被访问过
            visited.append(moveStr) #把节点添加到已访问的列表中
            if moveNode.x == target_x and moveNode.y == target_y: #如果当前节点是目标节点 则返回当前节点的路径
                return moveNode.w #返回当前节点的路径 等于直接发送让小车往哪边走的指令
            if moveNode.x < m - 1: #检测当前点是否在地图边界范围呢
                if map_int[moveNode.y][moveNode.x + 1] == 0:#检测右边是否为通路
                    queue.append(down(moveNode)) #把右边那一点添加到新队列里
            if moveNode.y > 0: #检测当前节点y坐标是否大于0
                if map_int[moveNode.y - 1][moveNode.x] == 0:#检测上方位置是否为0 如果是0 则可以移动
                    queue.append(left(moveNode))
            if moveNode.y < n - 1: #类比上面
                if map_int[moveNode.y + 1][moveNode.x] == 0:
                    queue.append(right(moveNode))
            if moveNode.x > 0: #类比上面
                if map_int[moveNode.y][moveNode.x - 1] == 0:
                    queue.append(up(moveNode))


def take_point(start_x, start_y, target_x, target_y, point_list, map_int):
    num = len(point_list) #计算识别到的圆的个数
    print("宝藏点:",point_list,num)
    best_way = '' #最短的路径
    best_way_point = () #最短路径点的顺序
    min_step = 999999999 #最短路径的步数
    case_n = 0 #定义每一种排序的方法
    for case in itertools.permutations(point_list, num): #列出8个宝藏点所有的情况 排列组合
        case_n += 1 #每次把排列的组合加一
        total_way = '' #定义总长度
        cnt = 0  #表示8个宝藏点里面的某个单一宝藏点
        for liter_point in case: #在每一次排列组合里面遍历宝藏点
            cnt = cnt + 1 #自加1
            step_way = '' #初始化
            if cnt == 1: #如果找的是第一个宝藏点
                step_way = get_way(start_x, start_y, liter_point[0], liter_point[1], map_int) #找出起始点到第一个宝藏点之间的距离
            elif 1 < cnt <= num: #如果是中间几个宝藏点 计算上一宝藏点到下一宝藏点之间的距离
                step_way = get_way(last_point[0], last_point[1], liter_point[0], liter_point[1], map_int)
            last_point = liter_point #把上一次宝藏点的值赋值给下一宝藏点的值
            step_way += '#' #找到一个宝藏点与宝藏点之间的分割
            total_way += step_way #把总步数相加
            if cnt == num: #如果开始找最后一个宝藏点 找出最后一个宝藏点到终点的距离
                end_way = get_way(liter_point[0], liter_point[1], target_x, target_y, map_int)
                total_way += end_way
        if len(total_way) < min_step:#如果总长度小于最小步数 替换
            min_step = len(total_way)
            best_way = total_way
            best_way_point = case
        print(case_n)
    return best_way, best_way_point #返回最优解 最优遍历方式


def simulate(start_x, start_y, way, point_xy, map_int):#迷宫循迹可视化
    plt.figure(2)  #创建一个图表
    plt.clf() #清楚当前图像的内容
    plt.imshow(map_int, cmap="Set3") #绘制迷宫地图
    px, py = zip(*point_xy) #把有宝藏点的坐标转化为特殊的坐标
    '''
    mngr = plt.get_current_fig_manager()  # 获取当前figure manager
    mngr.window.wm_geometry("+900+200")  # 调整窗口在屏幕上弹出的位置
    '''
    plt.scatter(px, py, marker='+', color='coral') #设置宝藏点的坐标用什么图形标记 颜色
    robot = [start_x, start_y] #显示当前坐标
    cnt = 0 #记录找到的宝藏点
    for i in range(len(way)):
        last_robot = robot.copy() #复制当前机器人的坐标
        step = way[i] #赋值小车如何移动的方向
        if step == 'L': #机器人的路径方向
            robot[0] = robot[0] - 1
        elif step == 'R':
            robot[0] = robot[0] + 1
        elif step == 'U':
            robot[1] = robot[1] - 1
        elif step == 'D':
            robot[1] = robot[1] + 1
        elif step == '#':
            cnt = cnt + 1 #寻找到的宝藏点加一
            plt.text(robot[0], robot[1], str(cnt), fontsize=10)  #绘制文本
            pass
        draw = [last_robot, robot] #绘制机器人上一步和当前的步数
        dx, dy = zip(*draw) #绘制路径
        plt.plot(dx, dy, color='blue')
    plt.pause(0.1) #暂停0.1s


# sta_x = 20
# sta_y = 1
# tar_x = 0
# tar_y = 19
# map_i = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#          [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
#          [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
#          [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
#          [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
#          [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
#          [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
#          [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
#          [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
#          [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
#          [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
#          [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
#          [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
#          [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
#          [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
#          [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
#          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
#
# # po_list = [[1, 5], [3, 9], [11, 7], [15, 5], [19, 15], [17, 11], [9, 13], [5, 15]]
# po_list_t = [[1, 5], [3, 9], [11, 7], [15, 5], [19, 15], [17, 11]]
#
# for point in po_list_t:
#     if map_i[point[1]][point[0]] == 1:
#         print("find wrong:")
#         print(point)
# best_w, best_p = take_point(sta_x, sta_y, tar_x, tar_y, po_list_t, map_i)
# print(best_w)
# print(len(best_w) - len(po_list_t))
# print(best_p)
# T2 = time.time()
# print('程序运行时间:%s毫秒' % ((T2 - T1) * 1000))
# simulate(sta_x, sta_y, best_w, po_list_t, map_i)
