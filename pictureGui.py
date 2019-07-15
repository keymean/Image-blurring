# -*- coding: utf-8 -*-
"""
Created on Mon May 14 15:10:27 2018

@author: DELL
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tkinter import *
import tkinter.messagebox as messagebox
from PIL import Image, ImageTk, ImageFilter
from tkinter.filedialog import askopenfile
import numpy as np
import math
from scipy.misc import toimage 
from scipy import signal 
#import cv2
class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.path = None
        self.tkImage = None
        self.frame_left = Frame(self.master, height=800, width=450)
        self.frame_left.pack_propagate(0)
        self.frame_left.pack(side='left')
        self.frame_right = Frame(self.master, height=800, width=350)
        self.frame_right.pack_propagate(0)
        self.frame_right.pack(side='right')
        self.frame_right_t = Frame(self.frame_right, height=100, width=350)
        self.frame_right_t.pack_propagate(0)
        self.frame_right_t.pack(side='top')
        self.frame_right_b = Frame(self.frame_right, height=600, width=350)
        self.frame_right_b.pack_propagate(0)
        self.frame_right_b.pack()
        Label(self.frame_right_b, text='参数设置').pack()
        self.frame_right_b_t = Frame(self.frame_right_b, height=50, width=300)
        self.frame_right_b_t.pack_propagate(0)
        self.frame_right_b_t.pack(side='top')
        self.frame_right_b_avg = Frame(self.frame_right_b, height=50, width=300)
        self.frame_right_b_avg.pack_propagate(0)
        self.frame_right_b_avg.pack(side='top')
        self.frame_right_b_m = Frame(self.frame_right_b, height=50, width=300)
        self.frame_right_b_m.pack_propagate(0)
        self.frame_right_b_m.pack(side='top')
        self.frame_right_b_b = Frame(self.frame_right_b, height=50, width=300)
        self.frame_right_b_b.pack_propagate(0)
        self.frame_right_b_b.pack(side='top')
        self.createWidgets()
    def createWidgets(self):
        Label(self.frame_left, text='原始图片（上），处理图片（下）').pack()
        self.label1 = Label(self.frame_left)
        self.label1.pack()  #原始图片
        self.label2 = Label(self.frame_left)
        self.label2.pack()  #处理图片
        Label(self.frame_right_t, text='选择图片').pack()
        Label(self.frame_right_t, text='目标路径').pack(side='left')
        self.pathVar = StringVar()
        self.entry1 = Entry(self.frame_right_t, textvariable=self.pathVar).pack(side='left')
        self.button1 = Button(self.frame_right_t, text='路径选择', command=self.selectPath).pack(side='left')
        #Label(self.frame_right_b, text='参数设置').pack()
        self.gaussian_num = StringVar()
        self.gaussian_num.set(1)
        Label(self.frame_right_b_t, text='高斯平滑处理：方差').pack(side='left')
        self.entry2 = Entry(self.frame_right_b_t,textvariable=self.gaussian_num).pack(side='left')
        #self.entry2.insert(10,1)
        self.button2 = Button(self.frame_right_b_t, text='确定', command=self.Gaussian).pack(side='left')
        Label(self.frame_right_b_avg, text="均值滤波:模板大小").pack(side='left')
        self.average_num = StringVar()
        self.average_num.set(3)
        Entry(self.frame_right_b_avg, textvariable=self.average_num).pack(side='left')
        Button(self.frame_right_b_avg, text="确定", command=self.Average_filtering).pack(side='left')
        Label(self.frame_right_b_m, text='直方图均化').pack(side='left')
        self.button3 = Button(self.frame_right_b_m, text='确定', command=self.histogram_equalization).pack(side='left')
        Label(self.frame_right_b_b, text='拉普拉斯滤波').pack(side='left')
        self.button4 = Button(self.frame_right_b_b, text='确定', command=self.Laplace_filtering).pack(side='left')
    def hello(self):
        name = self.nameInput.get() or 'world'
        messagebox.showinfo('Message', 'Hello, %s' % name)
    def selectPath(self):
        self.path = askopenfile(filetypes=[('图片.','jpg'),('图片.','bmp'),('图片.','png')])
        if self.path != None:
              self.pathVar.set(self.path.name)
              self.pilImage = self.picture_shrink()
              self.tkImage = ImageTk.PhotoImage(image=self.pilImage)
              self.label1.config(image=self.tkImage)
    def Gaussian_step(self, image):
        PI = 3.1415926
        w,h = image.size
        nr = np.zeros((w,h))
        ng = np.zeros((w,h))
        nb = np.zeros((w,h))
        for i in range(0,w):
            for j in range(0,h):
                nr[i][j] = image.getpixel((i,j))[0]
                ng[i][j] = image.getpixel((i,j))[1]
                nb[i][j] = image.getpixel((i,j))[2]
        summat = 0
        r = int(self.gaussian_num.get())
        ma = np.empty((2*r+1,2*r+1))
        for i in range(0,2*r+1):
            for j in range(0,2*r+1):
                gaussp = (1/(2*PI*(r**2))) * math.e**(-((i-r)**2+(j-r)**2)/(2*(r**2))) 
                ma[i][j] = gaussp
                summat += gaussp
        for i in range(0,2*r+1):
            for j in range(0,2*r+1):
                ma[i][j] = ma[i][j]/summat
        newr = np.zeros((w,h))
        newg = np.zeros((w,h))
        newb = np.zeros((w,h))
        for i in range(r+1,w-r):
            for j in range(r+1,h-r):
                o = 0 
                for x in range(i-r,i+r+1):
                    p = 0
                    for y in range(j-r,j+r+1):
                        #print("x{},y{},o{},p{}".format(x,y,o,p))
                        newr[i][j] += nr[x][y]*ma[o][p]
                        newg[i][j] += ng[x][y]*ma[o][p]
                        newb[i][j] += nb[x][y]*ma[o][p]
                        p += 1
                    o += 1    
        for i in range(r+1,w-r+1):
            for j in range(r+1,h-r+1):
                image.putpixel((i,j),(int(newr[i][j]),int(newg[i][j]),int(newb[i][j])))
        return image
    def Gaussian_step1(self, image):
        PI = 3.1415926
        w,h = image.size
        nr, ng, nb = self.resolve_rgb(image)
        summat = 0
        r = int(self.gaussian_num.get())
        ma = np.zeros((2*r+1,2*r+1))
        for i in range(0,2*r+1):
            for j in range(0,2*r+1):
                gaussp = (1/(2*PI*(r**2))) * math.e**(-((i-r)**2+(j-r)**2)/(2*(r**2))) 
                ma[i][j] = gaussp
                summat += gaussp
        for i in range(0,2*r+1):
            for j in range(0,2*r+1):
                ma[i][j] = ma[i][j]/summat
        newr = np.zeros((h,w))
        newg = np.zeros((h,w))
        newb = np.zeros((h,w))
        for i in range(r+1,h-r):
            for j in range(r+1,w-r):
                o = 0 
                for x in range(i-r,i+r+1):
                    p = 0
                    for y in range(j-r,j+r+1):
                        #print("x{},y{},o{},p{}".format(x,y,o,p))
                        newr[i][j] += nr[x][y]*ma[o][p]
                        newg[i][j] += ng[x][y]*ma[o][p]
                        newb[i][j] += nb[x][y]*ma[o][p]
                        p += 1
                    o += 1 
        imaged = self.group_rgb(newr, newg, newb)
        imaged = self.fill_edge(r+2,image,imaged)
        return imaged
    def Gaussian(self):
        image_crop = self.pilImage.crop()
        image_crop_Gaussian = self.Gaussian_step1(image_crop)
        self.tkImage_Gaussian = ImageTk.PhotoImage(image=image_crop_Gaussian)
        self.label2.config(image=self.tkImage_Gaussian)
    def Average_filtering_step(self,image):
        arr_im_rcolor,arr_im_gcolor,arr_im_bcolor = self.resolve_rgb(image)
        matrix_size = int(self.average_num.get())
        ave = 1/(matrix_size**2)
        average_matrix = np.zeros((matrix_size, matrix_size))
        average_matrix.fill(ave)
        arr_im_rcolor = signal.convolve2d(arr_im_rcolor, average_matrix, mode='same')
        arr_im_gcolor = signal.convolve2d(arr_im_gcolor, average_matrix, mode='same')
        arr_im_bcolor = signal.convolve2d(arr_im_bcolor, average_matrix, mode='same')
        imaged = self.group_rgb(arr_im_rcolor, arr_im_gcolor, arr_im_bcolor)
        imaged = self.fill_edge(matrix_size, image, imaged)
        return imaged
    def fill_edge(self, size, image, imaged):
        image = np.array(image)
        imaged = np.array(imaged)
        h, w ,d = image.shape
        print("image:",image.shape)
        print("imaged:",imaged.shape)
        size = size-2
        for i in range(0,size):
            for j in range(0,w):
                imaged[i][j][0] = image[i][j][0]
                imaged[i][j][1] = image[i][j][1]
                imaged[i][j][2] = image[i][j][2]
                imaged[h-size-1+i][j][0] = image[h-size-1+i][j][0]
                imaged[h-size-1+i][j][1] = image[h-size-1+i][j][1]
                imaged[h-size-1+i][j][2] = image[h-size-1+i][j][2]
        for j in range(0,size):
            for i in range(0,h):
                imaged[i][j][0] = image[i][j][0]
                imaged[i][j][1] = image[i][j][1]
                imaged[i][j][2] = image[i][j][2]
                imaged[i][w-size+j][0] = image[i][w-size+j][0]
                imaged[i][w-size+j][1] = image[i][w-size+j][1]
                imaged[i][w-size+j][2] = image[i][w-size+j][2]
        for j in range(w-size,w):
            for i in range(0,h):
                imaged[i][j][0] = image[i][j][0]
                imaged[i][j][1] = image[i][j][1]
                imaged[i][j][2] = image[i][j][2]
        imaged = Image.fromarray(imaged)
        return imaged
    def Average_filtering(self):
        image_crop = self.pilImage.crop()
        image_crop_Average = self.Average_filtering_step(image_crop)
        self.tkImage_Average = ImageTk.PhotoImage(image=image_crop_Average)
        self.label2.config(image=self.tkImage_Average)
    def histogram_equalization_step(self, im_source):
        def histImageArr(im_arr, cdf):  
            cdf_min = cdf[0]  
            im_w = len(im_arr[0])  
            im_h = len(im_arr)  
            im_num = im_w*im_h  
            color_list = []  
            i=0
            # 通过累积分布函数计算灰度转换值  
            while i<256:  
                if i>len(cdf) - 1:  
                    color_list.append(color_list[i-1])  
                    break  
                tmp_v = (cdf[i] - cdf_min)*255/(im_num-cdf_min)  
                color_list.append(tmp_v)  
                i += 1  
            # 产生均衡化后的图像数据  
            arr_im_hist = []  
            for itemL in im_arr:  
                tmp_line = []  
                for item_p in itemL:  
                    tmp_line.append(color_list[item_p])  
                arr_im_hist.append(tmp_line)
            return arr_im_hist  
        def beautyImage(im_arr):  
            imhist, bins = np.histogram(im_arr.flatten(), range(256))  
            cdf = imhist.cumsum()  
          
            return histImageArr(im_arr, cdf) 
        # 这一部分是对彩图的直方图均衡化例子  
        arr_im_rgb  = np.array(im_source)  
        arr_im_rcolor = []  
        arr_im_gcolor = []  
        arr_im_bcolor = []  
        i = 0  
        # 分离三原色通道
        for itemL in arr_im_rgb:  
            arr_im_gcolor.append([])  
            arr_im_rcolor.append([])  
            arr_im_bcolor.append([])  
            for itemC in itemL:  
                arr_im_rcolor[i].append(itemC[0])  
                arr_im_gcolor[i].append(itemC[1])  
                arr_im_bcolor[i].append(itemC[2])  
            i = 1+i
        if True:  
            # 三个通道通过各自的分布函数来处理  
            arr_im_rcolor_hist = beautyImage(np.array(arr_im_rcolor))  
            arr_im_gcolor_hist = beautyImage(np.array(arr_im_gcolor))  
            arr_im_bcolor_hist = beautyImage(np.array(arr_im_bcolor))
        # 合并三个通道颜色到图片  
        i = 0  
        arr_im_hist = []  
        while i<len(arr_im_rcolor_hist):  
            ii = 0  
            tmp_line = []  
            while ii < len(arr_im_rcolor_hist[i]):  
                tmp_point = [arr_im_rcolor_hist[i][ii], arr_im_gcolor_hist[i][ii],arr_im_bcolor_hist[i][ii]]  
                tmp_line.append(tmp_point)  
                ii += 1  
            arr_im_hist.append(tmp_line)  
            i += 1
        im_beauty = toimage(np.array(arr_im_hist), 255) 
        return im_beauty
    def histogram_equalization(self):
        image_crop = self.pilImage.crop()
        image_crop_equalization = self.histogram_equalization_step(image_crop)
        self.tkImage_equalization = ImageTk.PhotoImage(image=image_crop_equalization)
        self.label2.config(image=self.tkImage_equalization)
    def Laplace_filtering_step(self, im_source):
        arr_im_rcolor,arr_im_gcolor,arr_im_bcolor = self.resolve_rgb(im_source)
        laplace_matrix = np.array([[0,1,0],[1,-4,1],[0,1,0]])
        arr_im_rcolor = signal.convolve2d(arr_im_rcolor, laplace_matrix, mode='same')
        arr_im_gcolor = signal.convolve2d(arr_im_gcolor, laplace_matrix, mode='same')
        arr_im_bcolor = signal.convolve2d(arr_im_bcolor, laplace_matrix, mode='same')
        image = self.group_rgb(arr_im_rcolor, arr_im_gcolor, arr_im_bcolor)
        return image
    def resolve_rgb(self, image):
        w, h = image.size
        arr_image = np.array(image)
        arr_im_rcolor = np.zeros((h, w))
        arr_im_gcolor = np.zeros((h, w))
        arr_im_bcolor = np.zeros((h, w))
        for i in range(0,h):
            for j in range(0,w):
                arr_im_rcolor[i][j] = arr_image[i][j][0]
                arr_im_gcolor[i][j] = arr_image[i][j][1]
                arr_im_bcolor[i][j] = arr_image[i][j][2]
        return arr_im_rcolor,arr_im_gcolor,arr_im_bcolor
    def group_rgb(self,r,g,b):
        w, h = r.shape
        arr_image = np.zeros((w, h, 3))
        for i in range(0,w):
            for j in range(0,h):
                arr_image[i][j][0] = r[i][j]
                arr_image[i][j][1] = g[i][j]
                arr_image[i][j][2] = b[i][j]
        arr_image = arr_image.astype(np.uint8)
        return Image.fromarray(arr_image)
    def Laplace_filtering(self):
        image_crop = self.pilImage.crop()
        image_crop_laplace = self.Laplace_filtering_step(image_crop)
        self.tkImage_laplace = ImageTk.PhotoImage(image=image_crop_laplace)
        self.label2.config(image=self.tkImage_laplace)
    def picture_shrink(self):
        image = Image.open(self.path.name)
        w, h = image.size
        if w >500 or h > 350:
            image.thumbnail((w/(w/500), h/(h/350)))
        return image
def text(im_source):
    # 这一部分是对彩图的直方图均衡化例子  
    arr_im_rgb  = np.array(im_source)  
    arr_im_rcolor = []  
    arr_im_gcolor = []  
    arr_im_bcolor = []  
    i = 0  
    # 分离三原色通道
    for itemL in arr_im_rgb:  
        arr_im_gcolor.append([])  
        arr_im_rcolor.append([])  
        arr_im_bcolor.append([])  
        for itemC in itemL:  
            arr_im_rcolor[i].append(itemC[0])  
            arr_im_gcolor[i].append(itemC[1])  
            arr_im_bcolor[i].append(itemC[2])  
        i = 1+i
    return arr_im_rcolor,arr_im_gcolor,arr_im_bcolor
if __name__ == "__main__":
    root = Tk()
    app = Application(root)
    # 设置窗口标题:
    app.master.title('图像处理')
    app.master.geometry('800x700')
    app.master.resizable(width=True, height=True)
    # 主消息循环:
    app.mainloop()
