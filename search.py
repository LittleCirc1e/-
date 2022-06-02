from PIL import ImageTk
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from method import *
import os
import asyncio
from on_the_fly_detect import *

class search(object):
    def __init__(self, master=None):
        self.app = master
        self.app.geometry("1280x720")
        # 启动后创建组件
        self.create()
        #self.model =

    def create(self):
        # 创建一个输入框
        img_path = tk.Entry(self.app, font=("宋体", 18), )
        # 顺序布局
        img_path.pack()
        # 坐标
        img_path.place(relx=0.2, rely=0.3, relwidth=0.6, relheight=0.1)

        # 参数是：要适应的窗口宽、高、Image.open后的图片
        # 调整尺寸
        def img_resize(w_box, h_box, pil_image):
            print(pil_image)
            # 获取图像的原始大小
            width, height = pil_image.size
            f1 = 1.0 * w_box / width
            f2 = 1.0 * h_box / height
            factor = min([f1, f2])
            width = int(width * factor)
            height = int(height * factor)
            # 更改图片尺寸，Image.ANTIALIAS：高质量
            return pil_image.resize((width, height), Image.ANTIALIAS)

        #  添加搜索图标
        img_s = Image.open('icon/search.png')
        img_s_resized = img_resize(0.05 * 1280, 0.1 * 720, img_s)
        self.img_s = ImageTk.PhotoImage(img_s_resized)
        lbs = tk.Label(self.app, imag=self.img_s, compound=tk.CENTER, bg='white')
        lbs.place(relx=0.15, rely=0.3, relwidth=0.05, relheight=0.1)

        #  添加图片图标
        img_t = Image.open('icon/picture.png')
        img_t_resized = img_resize(0.05 * 1280, 0.1 * 720, img_t)
        self.img_t = ImageTk.PhotoImage(img_t_resized)
        lbt = tk.Label(self.app, imag=self.img_t, compound=tk.CENTER, bg='#6888a8')
        lbt.place(relx=0.05, rely=0.05, relwidth=0.05, relheight=0.1)

        #  本地上传图标
        upload = tk.Button(self.app, text="本地上传", font=("宋体", 20), command=lambda: img_choose(img_path))
        upload.place(relx=0.8, rely=0.3, relwidth=0.1, relheight=0.1)

        #  综合最优
        enter_color = tk.Button(self.app, text="综合最优", font=("宋体", 20), command=lambda: enter(0))
        enter_color.place(relx=0.4, rely=0.45, relwidth=0.2, relheight=0.1)

        #  选择视频
        def img_choose(img_path):
            # 打开文件管理器，选择图片
            self.app.picture = filedialog.askopenfilename(parent=self.app, initialdir=os.getcwd(), title="本地上传")
            # 同时将图片路径写入行内
            # img_path.delete(0,"end")
            img_path.insert(0, self.app.picture)
            # img_path[0] = self.app.picture

        #  根据输入框地址进行图像检索
        def enter(option):
            # 被检索的图像路径
            search_path = img_path.get()

            # 未选择图片，则不检索
            #if (search_path == ''):
                #return

            # 计算检索的耗时
            # 0代表使用残差attention进行搜索
            if (option == 0):
                print(search_path)
                detector = Worker()
                result1 = detector.work(search_path)
                draw(result1[1])
                print(result1[0])
                print(result1[1])


            #  关闭主页面，创建结果界面
            self.app.destroy()
            result = tk.Tk()
            result.geometry("1280x720")
            result.title('检测结果')
            photo = tk.PhotoImage(file="icon/1.gif")  # 背景图片

            background = tk.Label(result, image=photo, compound=tk.CENTER)
            background.place(relx=0, rely=0, relwidth=1, relheight=1)

            backbutton = tk.Button(result, text="返回", font=("宋体", 25), command=lambda: back(result))
            backbutton.place(relx=0.8, rely=0.1, relwidth=0.08, relheight=0.08)

            word2 = tk.Label(result, text='检测结果：', font=("宋体", 25),  background="#f8caa8", compound=tk.CENTER)
            word2.place(relx=0.1, rely=0, relwidth=0.2, relheight=0.07)

            #  上传的图片
            img0 = Image.open("./2.jpg")
            img0_resized = img_resize(0.7 * 1280, 0.7 * 720, img0)
            img0 = ImageTk.PhotoImage(img0_resized)
            lb0 = tk.Label(result, image=img0, compound=tk.CENTER)
            lb0.place(relx=0.1, rely=0.1, relwidth=0.6, relheight=0.9)
            if (result1[0] == 0):
                word3 = tk.Label(result, text="鉴定为真", font=("宋体", 30), background="#00FF00", compound=tk.CENTER)
                word3.place(relx=0.75, rely=0.5, relwidth=0.2, relheight=0.07)
            else:
                word3 = tk.Label(result, text="鉴定为假", font=("宋体", 30), background="#FF0000", compound=tk.CENTER)
                word3.place(relx=0.75, rely=0.5, relwidth=0.2, relheight=0.07)
            result.mainloop()

        #  返回按键
        def back(result):
            # 摧毁当前结果页面
            result.destroy()
            #  创建主界面
            app = tk.Tk()
            app.title('MADD')
            background = tk.PhotoImage(file="icon/1.gif")  # 背景图片

            #  添加背景和标题
            bg = tk.Label(app, image=background, compound=tk.CENTER, bg="#989cb8")
            bg.place(relx=0, rely=0, relwidth=1, relheight=1)
            title = tk.Label(app, text='一种改进的基于多头注意力机制的deepfake检测算法', font=("宋体", 18),  background="#f8caa8", compound=tk.CENTER)
            title.place(relx=0.3, rely=0.2, relwidth=0.5, relheight=0.1)
            title = tk.Label(app, text='说明：使用了多种改进策略的deepfake检测算法\n'
                                       '包括两种改进attention的方法；\n'
                                       '两种选帧策略；\n'
                                       '以及三种增强纹理的滤波方法。', font=("宋体", 12), background="#f8caa8", compound=tk.CENTER)
            title.place(relx=0.3, rely=0.8, relwidth=0.4, relheight=0.2)
            search(app)
            app.mainloop()

