from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from io import BytesIO
from PIL import Image
import base64
import json
import PIL.Image
from torchvision import models, transforms
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
import torch
from skimage.transform import resize
import cv2
import requests

app = Flask(__name__)
CORS(app)

# Load the model from the .h5 file
model = load_model('C:\\Users\\lachl\\inkwell-ai\\flask-server\\model_with_background.h5')

# Load the pre-trained model
model_preprocessing = models.segmentation.deeplabv3_resnet101(pretrained=True)
model_preprocessing.eval()

# Define the preprocessing function
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225])
    ),
])

def crop_image(image, new_shape):
    current_shape = image.shape
    y_start = (current_shape[0] - new_shape[0]) // 2
    y_end = y_start + new_shape[0]
    x_start = (current_shape[1] - new_shape[1]) // 2
    x_end = x_start + new_shape[1]
    cropped_image = image[y_start:y_end, x_start:x_end, :]
    return cropped_image

def invert_colors(image):
    inverted_image = 255 - image
    return inverted_image

def overlay_images(background, overlay):
    # Create a copy of the background image to modify
    result = background.copy()

    # Resize the overlay image to match the background shape
    overlay_resized = overlay[:background.shape[0], :background.shape[1]]

    # Normalize the alpha values to range between 0 and 1
    overlay_alpha = overlay_resized[..., 3] / 255.0

    # Compute the weighted overlay using alpha blending
    result_alpha = 1 - (1 - overlay_alpha) * (1 - result[..., 3] / 255.0)
    result_alpha = np.clip(result_alpha, 0, 1)

    result[..., :3] = overlay_alpha[..., None] * overlay_resized[..., :3] + (1 - overlay_alpha[..., None]) * result[..., :3]
    result[..., 3] = result_alpha * 255

    return result

def convert_tattoo_to_rgba(image_array, alpha=255):
    # Check if the input image is already in RGBA format
    if image_array.shape[2] == 4:
        return image_array

    # Create an alpha channel with the specified value
    alpha_channel = np.full((image_array.shape[0], image_array.shape[1], 1), alpha, dtype=np.uint8)

    # Identify white pixels
    white_pixels = np.all(image_array == [1., 1., 1.], axis=-1)
    black_pixels = np.all(image_array == [0., 0., 0.], axis=-1)

    # Set alpha to 0 for white pixels
    alpha_channel[white_pixels] = 0
    alpha_channel[black_pixels] = int(0.8 * alpha)

    # Concatenate the alpha channel with the RGB image
    rgba_array = np.concatenate((image_array, alpha_channel), axis=2)

    return rgba_array


def load_image(final_img):
    img_array = np.array(final_img)
    img = Image.fromarray(img_array)
    with BytesIO() as buffer:
        img.save(buffer, format="PNG")
        png_data = buffer.getvalue()
    # Encode the PNG image data as a base64 string
    base64_str = base64.b64encode(png_data).decode('utf-8')
    # Encode the base64 string as a JSON object
    return json.dumps({"img": base64_str})


@app.route('/test', methods=['GET', 'POST'])
def make_app_images():
    return jsonify({"Abdul is a":"SK"})

@app.route('/upload', methods=['GET', 'POST'])
def make_image():
    print('hello')
    data = request.get_json()
    #arm_uri = data['armUri']  # Assume the arm image is now a URL
    arm_file_base64 = data['armFile']
    novaX = int(data['x'])
    novaY = int(data['y'])
    novaHeight = int(data['height'])
    novaWidth = int(data['width'])
    tattoo_uri = data['tattooUri']

    # Download the tattoo image
    from PIL import Image
    from io import BytesIO
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
    }

    # Send request with headers
    response = requests.get(tattoo_uri, headers=headers)

    # Open the image
    try:
        tattoo_img = Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Could not open image: {e}")

    tattoo_image = tattoo_img

    arm_file_data = base64.b64decode(arm_file_base64)
    img = Image.open(BytesIO(arm_file_data)).convert('RGB')
    img_np = np.array(img)

    new_height = 1280
    new_width = int(img_np.shape[1] / (img_np.shape[0] / new_height))
    resized_image = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    cropped_image = crop_image(resized_image, (1280, 720, 3))
    resized_pre_image = resize(cropped_image, (160, 90, 3), mode='constant')

    ## Predict keypoint
    from IPython.display import Image

    # add an extra dimension for the batch
    image_batch = np.expand_dims(resized_pre_image, axis=0)

    # Make predictions
    predictions = model.predict(image_batch)
    # Reshape the predicted keypoints array to a (4, 2) matrix
    predicted_keypoints = np.array(predictions).reshape((4, 2))
    # Read Tattoo Image
    from PIL import Image

    scaled_tattoo = tattoo_image.resize((int(novaWidth), int(novaHeight)))

    # Create a white background image to represent the arm
    background_color = (255, 255, 255, 0)  # White color in RGB
    background_image = Image.new("RGBA", (180, 320), background_color)

    # Overlay the image onto the canvas using PIL
    background_image.paste(scaled_tattoo, (novaX, novaY))

    # Convert the final image to a NumPy array
    image_array = np.array(background_image)
    # changing it to cv2
    resized_image = resize(image_array, (320, 180, 4), mode='constant')
    ## Warp Tattoo Image

    unnormalized_image = (resized_image * 255).astype(np.uint8)

    # Binarize the image array
    binarized_image_array = unnormalized_image

    # making the keypoints for the flat arm
    tl = (71, 137)
    bl = (63, 320)
    tr = (108, 137)
    br = (121, 320)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[predicted_keypoints[0][0], predicted_keypoints[0][1]], [predicted_keypoints[3][0], predicted_keypoints[3][1]], [predicted_keypoints[1][0], predicted_keypoints[1][1]], [predicted_keypoints[2][0], predicted_keypoints[2][1]]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(binarized_image_array, matrix, (720, 1280))
    def preprocess_and_predict(img):
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(img)

        # Resize the image while preserving the aspect ratio
        max_size = max(img_pil.size)
        aspect_ratio = img_pil.size[0] / img_pil.size[1]
        new_width = int(max_size * aspect_ratio)
        new_height = int(max_size)
        img_pil = img_pil.resize((new_width, new_height))
            
            # Preprocess the image
        img_tensor = preprocess(img_pil)

            # Create a mini-batch as expected by the model
        input_batch = img_tensor.unsqueeze(0)

            # Move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model_preprocessing.to('cuda')

            # Get the model prediction
        with torch.no_grad():
            output = model_preprocessing(input_batch)['out'][0]
        output_predictions = output.argmax(0)

            # Resize the mask to match the original image dimensions
        resized_mask = torch.nn.functional.interpolate(output_predictions.unsqueeze(0).unsqueeze(0).float(), size=(img_pil.height, img_pil.width), mode="nearest").squeeze(0).squeeze(0)

            # Create a binary mask
        human_mask = resized_mask == 15

        return human_mask

    mask = preprocess_and_predict(cropped_image)

    def apply_mask(image_array, mask_array):
        # The mask should be binary (0 or 1), we can enforce this in case it isn't
        mask_array = mask_array > 0

        # Ensure the image has an alpha channel
        if image_array.shape[2] == 3:
            image_array = np.dstack([image_array, np.ones_like(image_array[:, :, 0])*255])
            
        # Where the mask is 0 (or False), set the alpha channel to 0
        image_array[mask_array == 0, 3] = 0

        return image_array
    def reduce_opacity(image_array):
        # Ensure the image has an alpha channel
        
        if image_array.shape[2] == 3:
            image_array = np.dstack([image_array, np.ones_like(image_array[:, :, 0])*255])
            
        # Reduce the alpha channel by 20%
        image_array[:, :, 3] = image_array[:, :, 3] * 0.77

        return image_array
    new = apply_mask(transformed_frame, mask)
    hello = reduce_opacity(new)
    rgba_arm = convert_tattoo_to_rgba(cropped_image)

    overlaid_image = overlay_images(rgba_arm, hello)
    return load_image(overlaid_image)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
    #app.run(debug=True)