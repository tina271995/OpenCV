import streamlit as st
import cv2


def remove_background(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding to the blurred image
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black image with the same shape as the original image
    mask = np.zeros(image.shape[:2], np.uint8)

    # Draw the largest contour on the mask
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], 0, (255, 255, 255), -1)

    # Bitwise AND the original image and the mask to obtain the final image with the background removed
    final = cv2.bitwise_and(image, image, mask=mask)

    return final

st.set_page_config(page_title="Remove Background from Image", page_icon=":clapper:", layout="wide")

file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file_buffer is not None:
    image = np.array(bytearray(file_buffer.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    result = remove_background(image)

    st.image(result, use_column_width=True)
