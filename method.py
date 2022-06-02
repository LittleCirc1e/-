from matplotlib import pyplot as plt
import matplotlib


def draw(y):
    # 设置中文
    matplotlib.rc('font', family='SimSun')
    '''
    #
    y = [0.9999758 , 0.99993944, 0.99999607, 0.99999785, 0.99989653,
       0.99998617, 0.99994564, 0.99988258, 0.99973339, 0.99973911,
       0.99824607, 0.99968362, 0.99972421, 0.99999475, 0.99995613,
       0.9992336 , 0.99928099, 0.99999321, 0.99957901, 0.99994648]
    '''
    # 设置组数
    d = 6  # 组距
    num_bins = 5

    # 设置图表大小
    plt.figure(figsize=(10, 5), dpi=80)

    # 设置x轴刻度
    #plt.xticks(range(min(y), max(y) + d, d))  # max(y)+d 的目的是可以使得max(y)最终能取得到

    # 绘制直方图，第一个参数是总的数据，第二个参数是组数
    plt.hist(y, num_bins)

    # 绘制栅格
    plt.grid(alpha=0.3)

    # 设置x轴label
    plt.xlabel("置信度")

    # 设置y轴label
    plt.ylabel("帧数")

    # 设置标题title
    plt.title("检测视频帧的置信度统计")

    # 保存直方图
    plt.savefig("./2.jpg")


