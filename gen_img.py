# @Time    : 2019/2/22 17:05
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : gen_img.py
import string
from PIL import Image, ImageFont, ImageDraw
import random
import  os


digits=string.digits

text = "hello"
path='D:\project\data-set\digits2'

def gen_img(num=10):
    for i in range(num):
        text=''.join(random.sample(digits, 1))
        img_name='{}_{}.png'.format(''.join(random.sample(digits,4)),text)
        img_path=os.path.join(path,img_name)
        im = Image.new("RGB", (160, 60), (255, 255, 255))
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype('‪C:\Windows\Fonts\CALIST.TTF',42)
        dr.text((10, 5), text, font=font, fill="#000000")
        im.save(img_path)


if __name__ == '__main__':
    gen_img(3000)