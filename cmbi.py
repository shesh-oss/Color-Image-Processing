from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def full_process_image(filepath):
    image = cv2.imread(filepath)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(image_rgb)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Red color mask
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(image_hsv, lower_red, upper_red)
    red_filtered = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_red)

    # Edge detection on green channel
    edges_g = cv2.Canny(g, 100, 200)

    # Plot all in grid
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs[0, 0].imshow(image_rgb), axs[0, 0].set_title("Original RGB")
    axs[0, 1].imshow(image_hsv), axs[0, 1].set_title("HSV")
    axs[0, 2].imshow(image_lab), axs[0, 2].set_title("Lab")
    axs[1, 0].imshow(image_ycrcb), axs[1, 0].set_title("YCrCb")
    axs[1, 1].imshow(image_gray, cmap='gray'), axs[1, 1].set_title("Grayscale")
    axs[1, 2].imshow(red_filtered), axs[1, 2].set_title("Red Filtered")
    axs[2, 0].imshow(r, cmap='Reds'), axs[2, 0].set_title("Red Channel")
    axs[2, 1].imshow(g, cmap='Greens'), axs[2, 1].set_title("Green Channel")
    axs[2, 2].imshow(edges_g, cmap='gray'), axs[2, 2].set_title("Edges Detected")

    for ax in axs.ravel():
        ax.axis('off')

    plt.tight_layout()
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
    plt.savefig(result_path)
    plt.close()
    return 'result.png'

@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return "No image selected."

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
        file.save(filepath)

        result_filename = full_process_image(filepath)
        if result_filename:
            result_image = f'output/{result_filename}'
        else:
            return "Error processing image."

    return render_template('index.html', result_image=result_image)

@app.route('/static/output/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
