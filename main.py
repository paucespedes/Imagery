import cv2
import numpy as np

def save_difference_image(image1_path, image2_path, output_path):
    # Read the input images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Ensure the images have the same dimensions
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Compute the absolute difference between the images
    diff = cv2.absdiff(img1, img2)

    # Save the difference image
    cv2.imwrite(output_path, diff)

def compare_videos(original_path, noisy_path):
    # Open the video files
    original_cap = cv2.VideoCapture(original_path)
    noisy_cap = cv2.VideoCapture(noisy_path)

    # Check if videos opened successfully
    if not (original_cap.isOpened() and noisy_cap.isOpened()):
        print("Error: Couldn't open one or both video files.")
        return

    # Initialize lists to store mean and std deviation for each channel
    mean_diff_r_list, std_dev_diff_r_list = [], []
    mean_diff_g_list, std_dev_diff_g_list = [], []
    mean_diff_b_list, std_dev_diff_b_list = [], []

    while True:
        # Read frames from the videos
        ret1, original_frame = original_cap.read()
        ret2, noisy_frame = noisy_cap.read()

        # Break the loop if either video has reached the end
        if not (ret1 and ret2):
            break

        # Convert frames to grayscale for better comparison
        original_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        noisy_gray = cv2.cvtColor(noisy_frame, cv2.COLOR_BGR2GRAY)

        # Calculate the absolute pixel intensity differences
        diff_gray = cv2.absdiff(original_gray, noisy_gray)

        # Calculate mean and standard deviation of differences
        mean_diff = np.mean(diff_gray)
        std_dev_diff = np.std(diff_gray)

        # Print or store the results
        print(
            f"Frame {original_cap.get(cv2.CAP_PROP_POS_FRAMES)}: Mean Diff = {mean_diff}, Std Dev Diff = {std_dev_diff}")

        # Calculate the absolute pixel intensity differences for each color channel
        diff = cv2.absdiff(original_frame, noisy_frame)

        # Calculate mean and standard deviation for each channel
        mean_diff_r = np.mean(diff[:, :, 0])
        std_dev_diff_r = np.std(diff[:, :, 0])

        mean_diff_g = np.mean(diff[:, :, 1])
        std_dev_diff_g = np.std(diff[:, :, 1])

        mean_diff_b = np.mean(diff[:, :, 2])
        std_dev_diff_b = np.std(diff[:, :, 2])

        # Append results to lists
        mean_diff_r_list.append(mean_diff_r)
        std_dev_diff_r_list.append(std_dev_diff_r)

        mean_diff_g_list.append(mean_diff_g)
        std_dev_diff_g_list.append(std_dev_diff_g)

        mean_diff_b_list.append(mean_diff_b)
        std_dev_diff_b_list.append(std_dev_diff_b)

        # Print or store the results
        print(f"Frame {original_cap.get(cv2.CAP_PROP_POS_FRAMES)}: "
              f"Mean Diff (R) = {mean_diff_r}, Std Dev Diff (R) = {std_dev_diff_r}, "
              f"Mean Diff (G) = {mean_diff_g}, Std Dev Diff (G) = {std_dev_diff_g}, "
              f"Mean Diff (B) = {mean_diff_b}, Std Dev Diff (B) = {std_dev_diff_b}\n")

    # Calculate overall mean and std deviation for each channel
    overall_mean_r = np.mean(mean_diff_r_list)
    overall_std_dev_r = np.mean(std_dev_diff_r_list)

    overall_mean_g = np.mean(mean_diff_g_list)
    overall_std_dev_g = np.mean(std_dev_diff_g_list)

    overall_mean_b = np.mean(mean_diff_b_list)
    overall_std_dev_b = np.mean(std_dev_diff_b_list)

    # Print overall results
    print("\nOVERALL RESULTS:")
    print(f"Overall Mean Diff (R) = {overall_mean_r}, Overall Std Dev Diff (R) = {overall_std_dev_r}")
    print(f"Overall Mean Diff (G) = {overall_mean_g}, Overall Std Dev Diff (G) = {overall_std_dev_g}")
    print(f"Overall Mean Diff (B) = {overall_mean_b}, Overall Std Dev Diff (B) = {overall_std_dev_b}")

    # Determine which channel has the highest standard deviation
    channels = ['Red', 'Green', 'Blue']
    max_std_dev_channel = channels[np.argmax([overall_std_dev_r, overall_std_dev_g, overall_std_dev_b])]
    print(f"\nThe channel with the highest standard deviation is: {max_std_dev_channel}")

    # Release the video capture objects
    original_cap.release()
    noisy_cap.release()

if __name__ == "__main__":
    image1_path = "/home/pau/TFG/isolated-noises/Training-Step-0/Image-0/Frame-1/noise.png"
    image2_path = "/home/pau/TFG/isolated-noises/Training-Step-0/Image-0/Frame-2/noise.png"
    output_path = "/home/pau/TFG/isolated-noises/Training-Step-0/Image-0/Diff-2-1/diff.png"

    original_path = "/home/pau/TFG/new-train-dataset/original/bear.mp4"
    noisy_path = "/home/pau/TFG/new-train-dataset/noisy_videos/bear.mp4"

    compare_videos(original_path, noisy_path)

    save_difference_image(image1_path, image2_path, output_path)
