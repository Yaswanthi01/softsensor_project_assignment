#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
# import opencv2
import numpy as np 
import PIL
from PIL import Image
import glob
from PIL import ImageEnhance
import cv2
from threading import Thread


# In[3]:


# path = x_ray_images

image_list = []
for filename in glob.glob('x_ray_images/*.jpg'):
    im=Image.open(filename)
    image_list.append(im)


# In[4]:


img_count = len(image_list)
print(img_count)


# In[5]:


print(image_list)


# In[6]:


image_paths = list(glob.glob('x_ray_images/*'))
# PIL.Image.open(str(image_paths))

print(image_paths)


# In[7]:


PIL.Image.open(str(image_paths[0]))


# In[8]:



#Resizing the image 
#using this , the image is resized according to the required dimensions , in our case I have scaled down the image by 25%
#of the original size 

image= cv2.imread(str(image_paths[0]))
scale_percent =25 #percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# print(dim)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Resize", resized)
cv2.waitKey(0)


# In[9]:


#converting image to grayscale 

#the color image is converted to grayscale 

gray_image= cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
# cv2.namedWindow('Gray Image', cv2.WINDOW_NORMAL)
cv2.imshow("Gray Image", gray_image)
cv2.waitKey(0)


# In[10]:


#denoising an image using canny edge detection 
#noise refers to the presense of any unwanted or random variation in the brightness of the image 
#or any random variation in the color information. 

#using canny edge detection , we can minimize unwanted noise and detection of edges is performed .
#the canny() function consists of 3 attrbutes - the input image , threshol1 , threshold 2 
#  threshold 1 - all pixels below this value is not an edge 
# threshold 2 - all pixels above this value is an edge 
#all pixel values between the threshold is classified as an edge and not an edge based on its neighbours


denoised_image = cv2.Canny(gray_image, 50,90 )
cv2.imshow("Edge",np.hstack((gray_image, denoised_image) ))
cv2.waitKey(0)


# In[11]:


#Gaussian Image Processing - gaussian blur

#blur basically reduces the sharoness of the image and gives a rather faded effect to the image .
#here we use gausian blur - which is a smoothening technique that helps to reduce noise in an image  

blurred = cv2.GaussianBlur(gray_image,(5,5),cv2.BORDER_DEFAULT)
 
# display input and output image
cv2.imshow("Gaussian Smoothing",np.hstack((gray_image, blurred)))
cv2.waitKey(0)


# In[12]:


#altering the brightness of the image 
#this is perfomred by increasing the rgb value of each pixel in the image 

img = PIL.Image.open(image_paths[0])
converter = ImageEnhance.Brightness(img)
img2 = converter.enhance(0.5)
img3 = converter.enhance(2)


# In[13]:


def display(im , t):
    im.show(title = t)


t1=Thread(target=display,args=(img,"original img"))
t1.start()
t2=Thread(target=display,args=(img2, "reduced brightness"))
t2.start()
t3 = Thread(target=display,args=(img3, "increased brightness"))
t3.start()


# In[20]:


#sharpening an image using a filter 

#here we take he kernel value as follows :

kernel1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen_image = cv2.filter2D( resized ,   -1,   kernel1)
cv2.imshow("sharp image " ,sharpen_image)
cv2.waitKey()


# In[21]:


#Emboss Filter
#it will seem to bold the image for a clearer vision 

emboss_kernel = kernel = np.array([[-2, -1, 0],[-1, 1, 1],[0, 1, 2]])
result_image = cv2.filter2D(resized, -1, kernel)
cv2.imshow("embossed image " ,result_image)
cv2.waitKey()


# In[23]:


#Sobel Filter
#it is a method of edge detection that looks at the variation in the intensity of gradient 
#and any massive difference is considered for identifying edges

kernel = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
 
sobel_image = cv2.filter2D(resized, -1, kernel)
cv2.imshow("sobel image " ,sobel_image)
cv2.waitKey()


# In[24]:


#Outline Edge Detection

kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
 
outline_img = cv2.filter2D(resized, -1, kernel)
cv2.imshow("outline image " ,outline_img)
cv2.waitKey()

conclusion : 

for images like x-rays and for our purpose in this case , it is preferable to use the follwing image processing techniques and filters :

1) resizing the image - makes the images user-friendly and easy to observe 
2) conversion to grayscale - converting to grayscale usually simplifies processes and reduces complexity 
3) canny edge detection - As the thresholds give good outputs and reduce noise . other edge detection methods like outline and                             sobel do not seem to give results to the mark 
4) sharpeninng filter - This gives a certain amount of clarity to the image and makes it easier for our model to identify the                           catheters and lines in the x-ray  
5) emboss filter - The emboss filter seems to enhance the edges and in a way makes it old which again permits easier                                identification of catheters and line in x-rays 

How can deep learning help to detect presence of catheters and lines in images and
which model you prefer?

By carefully using deep learning concepts like Convolutional neural networks and with the help of image processing techniques and filters , we can automate the porcess of idenifying catheters and lines in an X-rays .
Usually , this process requires the observation of doctors which gives way to human error and further more , takes much more time .
By autmating the process , more accurate and faster results may be obtained .

Since , we see the usage of images in our case , the most suitable model to solve our issue is "Convolutional Neural Network model"

by the usage of the processes of convolution , padding , and pooling , we can obtain accurate results .