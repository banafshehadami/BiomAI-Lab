import cv2
import os

crop_size = 128

def crop(image, filename, output_folder):
    """Crop the given image and save it to the output folder."""
    if image is None:
        print("Error: Could not read image file:", filename)
    else:
        # Get the dimensions of the image
        height, width, channels = image.shape

        # Calculate the center point of the image
        center_x = int(width / 2)
        center_y = int(height / 2)

        # Check if the image is too small to crop
        if height < crop_size or width < crop_size:
            print("Error: Image is too small to crop:", filename)
        else:
            # Calculate the coordinates of the top-left corner of the cropped image
            crop_x = center_x - int(crop_size / 2)
            crop_y = center_y - int(crop_size / 2)

            # Check if the crop area is within the bounds of the image
            if crop_x < 0 or crop_y < 0 or crop_x + crop_size > width or crop_y + crop_size > height:
                print("Error: Could not crop image file:", filename)
            else:
                # Crop the image to the desired size
                cropped_image = image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

                # Check if the cropped image is exactly 100x100 pixels
                if cropped_image.shape[0] == crop_size and cropped_image.shape[1] == crop_size:
                    # Save the cropped image to a file in the output folder
                    output_filename = os.path.join(output_folder, filename)
                    cv2.imwrite(output_filename, cropped_image)
                else:
                    print("Error: Could not crop image file:", filename)
