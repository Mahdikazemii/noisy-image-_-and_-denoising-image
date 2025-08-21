import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

def MSE(image1,image2):

  mse = np.mean(np.square(image1.astype(float) - image2.astype(float)))
  
  return mse


def PSNR(image1, image2, peak=255):

  mse = MSE(image1,image2)
  
  psnr = 10*np.log10(peak**2/mse)
  
  return psnr




import matplotlib.pyplot as plt

from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
image1 = np.random.randint(0,256,(1280,720,3))
image2 = np.random.randint(0,256,(1280,720,3))

plt.imshow(image1)

print(f"Skimage -> PSNR: {peak_signal_noise_ratio(image1,image2,data_range=255):.4f} | MSR: {mean_squared_error(image1,image2):.4f}")
print(f"My Func -> PSNR: {PSNR(image1,image2):.4f} | MSR: {MSE(image1,image2):.4f}")





from skimage.util import random_noise
import numpy as np

class noisy_system():
  def example(self,img,**kwargs):
   
    noisy_image = random_noise(img,**kwargs)
    noisy_image = np.uint8(noisy_image*255)
    return noisy_image

  def create_salt_and_pepper_noise(self,img,amount=0.05):

    img = img/255


    h = img.shape[0]
    w = img.shape[1]


    s = 0.5
    p = 0.5


    result = img.copy()


    salt = np.ceil(amount * img.size * s)
    vec = []
    for i in img.shape:
      vec.append(np.random.randint(0, i-1, int(salt)))

    for r, c  in zip(vec[0], vec[1]):
      result[r, c] = 1


    pepper = np.ceil(amount * img.size * p)
    vec = []
    for i in img.shape:
      vec.append(np.random.randint(0, i-1, int(salt)))
    for r, c in zip(vec[0], vec[1]):
      result[r, c] = 0


    result = np.uint8(result*255)

    return result

  def create_gaussian_noise(self,img,mean=0,var=0.01):

    img = img/255

    result = img.copy()

    gauss = np.random.normal(mean, var**0.5, img.shape)
    result = result + gauss
    result = np.clip(result, 0, 1)

    result = np.uint8(result*255)

    return result


import cv2
images = []
image_number=11

fig=plt.figure(figsize=(10,10))

for number in range(1,image_number):

  img_path=f"/content/drive/MyDrive/Dataset/Image{number}.png"
  noise_maker               = noisy_system()
  image                     = cv2.imread(img_path)
  image                     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  images.append(noise_maker.example(image.copy(),mode="s&p"))
  images.append(noise_maker.example(image.copy(),mode="gaussian"))
  images.append(noise_maker.create_salt_and_pepper_noise(image.copy()))
  images.append(noise_maker.create_gaussian_noise(image.copy()))

fig, axs = plt.subplots(10, 4, figsize=(12, 30))

for i, ax in enumerate(axs.flatten()):
    if i < len(images):
        ax.imshow(images[i],cmap="gray")
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()



def findAllNeighbors(padImg,small_window,big_window,h,w):
    smallWidth = small_window//2
    bigWidth = big_window//2

    neighbors = np.zeros((padImg.shape[0],padImg.shape[1],small_window,small_window))

    for i in range(bigWidth,bigWidth + h):
        for j in range(bigWidth,bigWidth + w):
            neighbors[i,j] = padImg[(i - smallWidth):(i + smallWidth + 1) , (j - smallWidth):(j + smallWidth + 1)]

    return neighbors

def evaluateNorm(pixelWindow, neighborWindow, Nw):

    Ip_Numerator,Z = 0,0

    for i in range(neighborWindow.shape[0]):
      for j in range(neighborWindow.shape[1]):

        q_window = neighborWindow[i,j]

        q_x,q_y = q_window.shape[0]//2,q_window.shape[1]//2

        Iq = q_window[q_x, q_y]

        w = np.exp(-1*((np.sum((pixelWindow - q_window)**2))/Nw))

        Ip_Numerator = Ip_Numerator + (w*Iq)
        Z = Z + w

    return Ip_Numerator/Z



class NLMeans():

  def example(self,img,**kwargs):
    denoised_image = cv2.fastNlMeansDenoising(img,**kwargs)
    return denoised_image

  def solve(self,img,h,small_window=7,big_window=21):


    padImg = np.pad(img,big_window//2,mode='reflect')

    return self.NLM(padImg,img,h,small_window,big_window)

  def NLM(self,padImg, img, h, small_window, big_window):
    Nw = (h**2)*(small_window**2)

    h,w = img.shape

    result = np.zeros(img.shape)

    bigWidth = big_window//2
    smallWidth = small_window//2

    neighbors = findAllNeighbors(padImg, small_window, big_window, h, w)

    for i in range(bigWidth, bigWidth + h):
        for j in range(bigWidth, bigWidth + w):
            pixelWindow = neighbors[i,j]

            neighborWindow = neighbors[(i - bigWidth):(i + bigWidth + 1) , (j - bigWidth):(j + bigWidth + 1)]

            Ip = evaluateNorm(pixelWindow, neighborWindow, Nw)


            result[i - bigWidth, j - bigWidth] = max(min(255, Ip), 0)

    return result

salt_and_paper_h =   36
gaussian_h =         18
image_number=11
images=[]
titles=['original image','salt and pepper noise','gaussian noise','original image','salt and pepper denoise','gaussian denoise']

for number in range(1,image_number,2):
  denoiser                    = NLMeans()
  image                       = cv2.imread(f"/content/drive/MyDrive/Dataset/Image{number}.png")
  image                       = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  image1=noise_maker.example(image.copy(),mode="s&p")
  image2= noise_maker.example(image.copy(),mode="gaussian")

  images.append(image)
  images.append(image1)
  images.append(image2)

  images.append(image)
  images.append(denoiser.example(image1.copy(),h=salt_and_paper_h))
  images.append(denoiser.example(image2.copy(),h=gaussian_h))


fig, axs = plt.subplots(10,3, figsize=(18,50))

t=0

for i, ax in enumerate(axs.flatten()):
    if t>5:
          t=0

    if i < len(images):

          ax.imshow(images[i],cmap="gray")
          ax.axis('off')
          ax.set_title(titles[t],)
    else:
        ax.axis('off')
    t+=1
plt.tight_layout()
plt.subplots_adjust(wspace=0.1,hspace=0.1)
plt.show()


plt.imshow(image)
im1=denoiser.solve(image1.copy(),h=salt_and_paper_h)
img2=denoiser.solve(image2.copy(),h=gaussian_h)

fig=plt.figure(figsize=(20,30))
plt.subplot(1,3,1)
plt.imshow(im1,cmap='grey')
plt.subplot(1,3,2)
plt.imshow(img2,cmap='grey')
plt.subplot(1,3,3)
plt.imshow(image,cmap='grey')
