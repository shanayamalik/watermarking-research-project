import numpy as np

dark_image_grey = np.dot(img_array, [0.2989, 0.5870, 0.1140])
dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(dark_image_grey))
fig, ax = plt.subplots(1,4,figsize=(20, 5))

ax[0].imshow(dark_image_grey, cmap='gray')

transformed_image = dark_image_grey_fourier
watermark = np.zeros_like(transformed_image)
grid_width = 103
d = 10000
xc = xy = watermark.shape[0] / 2
for i in range(watermark.shape[0]):
    for j in range(watermark.shape[1]):
        if (i % grid_width == 0 or j % grid_width == 0) and abs((i-xc)**2 + (j-xy)**2) > d:
            watermark[i,j] = 50000

transformed_image += watermark
ax[1].imshow(np.log(abs(transformed_image)), cmap='gray')

recovered_image = abs(np.fft.ifft2(transformed_image))
ax[2].imshow(recovered_image, cmap='gray')

retransformed_image = np.fft.fftshift(np.fft.fft2(recovered_image))
ax[3].imshow(np.log(abs(retransformed_image)), cmap='gray')
plt.show()