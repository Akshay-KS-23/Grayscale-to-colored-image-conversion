import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def load_network(prototxt_path, caffemodel_path):
    """
    Load the colorization network.
    """
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    points = np.load('E:/Colorization-of-B-W-images-and-videos-main/model/pts_in_hull.npy')
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [points.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net

def colorize_image(net, img_path):
    """
    Colorize a grayscale image.
    """
    image = cv2.imread(img_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50 # Subtract the mean value
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    return colorized
def compare_images(imageA, imageB, title):
    # Compute the Structural Similarity Index (SSIM) between the two images
    s = ssim(imageA, imageB, multichannel=True, win_size=5)
    
    # Print the dimensions of the images for debugging
    print("Original Image Dimensions:", imageA.shape)
    print("Colorized Image Dimensions:", imageB.shape)

    # Setup the figure
    fig = plt.figure(title)
    plt.suptitle(f"SSIM: {s:.2f}")

    # Show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # Show second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # Show the images
    plt.show()

# Now, call this function where you compare the original and colorized images
compare_images(original_img, colorized_img, "Comparison")

# Load the network
net = load_network('E:/Colorization-of-B-W-images-and-videos-main/model/colorization_deploy_v2.prototxt',
                   'E:/Colorization-of-B-W-images-and-videos-main/model/colorization_release_v2.caffemodel')

# Directories
grayscale_dir = 'E:\\Colorization-of-B-W-images-and-videos-main\\landscape Images\\gray'
color_dir = 'E:\\Colorization-of-B-W-images-and-videos-main\\landscape Images\\color'

# Process each image
for filename in os.listdir(grayscale_dir):
    gray_path = os.path.join(grayscale_dir, filename)
    color_path = os.path.join(color_dir, filename)
    
    if not os.path.exists(color_path):  # Skip if the color version doesn't exist
        continue

    # Colorize and compare
    colorized_img = colorize_image(net, gray_path)
    original_img = cv2.imread(color_path)
    
    compare_images(original_img, colorized_img, "Comparison")
