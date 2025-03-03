import os
import numpy as np
from PIL import Image


def convert_images_to_npy(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Load the JPEG image
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path)

            # Convert the image to a NumPy array
            image_array = np.array(image)

            # Save the NumPy array in ".npy" format
            npy_file_path = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}.npy"
            )
            np.save(npy_file_path, image_array)


if __name__ == "__main__":
    # Specify input and output directories
    input_directory = "/home/se1131/murnong_sem-segm/ASL Poly Instance Seg.v53i.coco-segmentation/test/images"
    output_directory = "/home/se1131/murnong_sem-segm/ASL Poly Instance Seg.v53i.coco-segmentation/test/images_npy"

    # Call the function
    convert_images_to_npy(input_directory, output_directory)
