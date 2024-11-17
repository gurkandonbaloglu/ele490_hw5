import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float32
import os

# Create output directory
OUTPUT_PATH = "output"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


# Utility to save images
def save_images(image, title, filename):
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    image_path = os.path.join(OUTPUT_PATH, f"{filename}.png")
    plt.savefig(image_path)
    plt.close()


# Question 1: Display g_L and g_H
def question_1():
    u, v = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
    sigma_values = [1, 10, 50, 100]

    for sigma in sigma_values:
        g_l = np.exp(-(u**2 + v**2) / (2 * sigma**2))
        g_h = 1 - g_l

        save_images(g_l, f"g_L[u,v] with σ={sigma}", f"g_l_sigma_{sigma}")
        save_images(g_h, f"g_H[u,v] with σ={sigma}", f"g_h_sigma_{sigma}")


# Question 2: Chirp signal and instantaneous frequency
def question_2():
    n, m = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
    r = np.sqrt(n**2 + m**2)
    t_nm = np.cos(np.pi * r**2 / 256)
    save_images(t_nm, "Chirp Signal t[n, m]", "chirp_signal")

    phi_r = np.pi * r**2 / 256
    finst = np.gradient(phi_r, axis=0)

    k_values = np.arange(0, 9)
    r_k = 16 * k_values
    valid_indices = 128 + r_k < 256
    r_k = r_k[valid_indices]
    finst_k = finst[128 + r_k, 128]

    print("Instantaneous frequency at r = 16k:", finst_k)


# Question 3: Filter chirp signal with g_L and g_H
def question_3():
    n, m = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
    r = np.sqrt(n**2 + m**2)
    t_nm = np.cos(np.pi * r**2 / 256)
    T = np.fft.fft2(t_nm)
    sigma_values = [1, 10, 50, 100]

    for sigma in sigma_values:
        u, v = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
        g_l = np.exp(-(u**2 + v**2) / (2 * sigma**2))
        g_h = 1 - g_l

        T_gl = T * g_l
        T_gh = T * g_h

        t_gl = np.abs(np.fft.ifft2(T_gl))
        t_gh = np.abs(np.fft.ifft2(T_gh))

        save_images(t_gl, f"Filtered t[n,m] with g_L, σ={sigma}", f"t_gl_sigma_{sigma}")
        save_images(t_gh, f"Filtered t[n,m] with g_H, σ={sigma}", f"t_gh_sigma_{sigma}")


# Question 5: Filter periodic signal
def question_5():
    P = 256
    n, m = np.meshgrid(np.arange(P), np.arange(P))
    r = np.sqrt(n**2 + m**2)
    t_nm = np.cos(np.pi * r**2 / 256)
    T = np.fft.fft2(t_nm)
    sigma_values = [1, 10, 50, 100]

    for sigma in sigma_values:
        u, v = np.meshgrid(np.arange(-P//2, P//2), np.arange(-P//2, P//2))
        g_l = np.exp(-(u**2 + v**2) / (2 * sigma**2))
        g_h = 1 - g_l

        T_gl = T * g_l
        T_gh = T * g_h

        t_gl = np.abs(np.fft.ifft2(T_gl))
        t_gh = np.abs(np.fft.ifft2(T_gh))

        save_images(t_gl, f"Periodic Filtered t[n,m] with g_L, σ={sigma}", f"t_gl_sigma_{sigma}_periodic")
        save_images(t_gh, f"Periodic Filtered t[n,m] with g_H, σ={sigma}", f"t_gh_sigma_{sigma}_periodic")


# Question 6: Filter Cameraman image
def question_6():
    cameraman_image = img_as_float32(io.imread("cameraman.tif"))
    F_cameraman = np.fft.fft2(cameraman_image)
    sigma_values = [1, 10, 50, 100]

    for sigma in sigma_values:
        u, v = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
        g_l = np.exp(-(u**2 + v**2) / (2 * sigma**2))
        g_h = 1 - g_l

        F_gl = F_cameraman * g_l
        F_gh = F_cameraman * g_h

        img_gl = np.abs(np.fft.ifft2(F_gl))
        img_gh = np.abs(np.fft.ifft2(F_gh))

        save_images(img_gl, f"Cameraman Filtered with g_L, σ={sigma}", f"cameraman_gl_sigma_{sigma}")
        save_images(img_gh, f"Cameraman Filtered with g_H, σ={sigma}", f"cameraman_gh_sigma_{sigma}")


# Question 7: Zero-padded filtering
def question_7():
    cameraman_image = img_as_float32(io.imread("cameraman.tif"))
    original_size = cameraman_image.shape
    padded_size = (original_size[0] * 2, original_size[1] * 2)
    padded_image = np.pad(cameraman_image, ((0, original_size[0]), (0, original_size[1])), mode="constant")
    F_padded = np.fft.fft2(padded_image)
    sigma_values = [1, 10, 50, 100]

    for sigma in sigma_values:
        u, v = np.meshgrid(np.arange(-padded_size[0]//2, padded_size[0]//2),
                           np.arange(-padded_size[1]//2, padded_size[1]//2))
        g_l = np.exp(-(u**2 + v**2) / (2 * sigma**2))
        g_h = 1 - g_l

        F_gl = F_padded * g_l
        F_gh = F_padded * g_h

        img_gl_padded = np.abs(np.fft.ifft2(F_gl))[:original_size[0], :original_size[1]]
        img_gh_padded = np.abs(np.fft.ifft2(F_gh))[:original_size[0], :original_size[1]]

        save_images(img_gl_padded, f"Zero-Padded Filtered with g_L, σ={sigma}", f"cameraman_gl_zero_padded_sigma_{sigma}")
        save_images(img_gh_padded, f"Zero-Padded Filtered with g_H, σ={sigma}", f"cameraman_gh_zero_padded_sigma_{sigma}")


# Main function
if __name__ == "__main__":
    # Uncomment the question(s) you want to execute
    #question_1()
    # question_2()
    # question_3()
    # question_5()
    # question_6()
    # question_7()
