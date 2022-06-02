
from search import *

#  创建主界面
app = tk.Tk()
app.title('MADD')
background = tk.PhotoImage(file="icon/1.gif")  # 背景图片

#  添加背景和标题
bg = tk.Label(app, image=background, compound=tk.CENTER,bg="#989cb8")
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
