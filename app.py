import gradio as gr
import cv2
import dlib
import numpy as np
import os

# Load the dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to extract eye measurements
def extract_eye_measurements(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        return {'error': 'No faces detected'}

    face = faces[0]  # Assuming one face
    landmarks = predictor(gray, face)

    # Indices for left and right eyes
    left_eye_indices = [36, 37, 38, 39, 40, 41]
    right_eye_indices = [42, 43, 44, 45, 46, 47]

    eye_measurements = {}

    for eye_name, eye_indices in zip(['left', 'right'], [left_eye_indices, right_eye_indices]):
        eye_landmarks = []
        for idx in eye_indices:
            x = landmarks.part(idx).x
            y = landmarks.part(idx).y
            eye_landmarks.append((x, y))

        # Convert to numpy array
        eye_landmarks = np.array(eye_landmarks)

        # Compute eye aspect ratio (EAR)
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        ear = (A + B) / (2.0 * C)

        # Compute width and height
        width = C
        height = (A + B) / 2.0

        # Determine eye shape (simplified)
        if ear > 0.3:
            shape = 'Round'
        else:
            shape = 'Almond'

        eye_measurements[eye_name] = {
            'landmarks': eye_landmarks,
            'aspect_ratio': ear,
            'width': width,
            'height': height,
            'shape': shape
        }

    return eye_measurements

# Function to recommend lashes based on measurements
def recommend_lashes(eye_measurements, style_preference='Natural'):
    lash_recommendations = {}
    for eye, data in eye_measurements.items():
        # Example recommendations based on eye shape
        shape = data['shape']
        if shape == 'Round':
            length = 8.0  # Shorter lashes
            curl = 'C'    # Medium curl
            volume = 'Light'
        elif shape == 'Almond':
            length = 10.0  # Longer lashes
            curl = 'D'     # Strong curl
            volume = 'Full'

        # Adjustments based on style preference
        if style_preference == 'Dramatic':
            length += 2.0
            volume = 'Extra Full'
        elif style_preference == 'Natural':
            pass  # Keep as is
        elif style_preference == 'Subtle':
            length -= 1.0
            volume = 'Light'

        lash_recommendations[eye] = {
            'length': length,
            'curl': curl,
            'volume': volume
        }

    return lash_recommendations

# Function to draw eye landmarks
def draw_eye_landmarks(image, landmarks):
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# Function to annotate eye information
def annotate_eye_info(image, eye, data):
    x = int(np.mean(data['landmarks'][:, 0]))
    y = int(np.min(data['landmarks'][:, 1])) - 10

    text = f"{eye.capitalize()} Eye: {data['shape']}"
    cv2.putText(image, text, (x - 50, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Function to annotate lash recommendations
def annotate_lash_info(image, eye, rec, data):
    x = int(np.mean(data['landmarks'][:, 0]))
    y = int(np.max(data['landmarks'][:, 1])) + 20

    text = f"Lash: Length {rec['length']}, Curl {rec['curl']}, {rec['volume']}"
    cv2.putText(image, text, (x - 50, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Function to select lash image based on recommendations
def select_lash_image(rec):
    # For simplicity, return a placeholder image path
    return 'lash_placeholder.png'

# Function to overlay lash image on the eye
def overlay_lash(image, eye_landmarks, lash_image_path):
    # Load the lash image with alpha channel
    lash_image = cv2.imread(lash_image_path, cv2.IMREAD_UNCHANGED)
    if lash_image is None:
        return  # If the lash image is not found, skip overlay

    # Compute the bounding box of the eye landmarks
    x_min = int(np.min(eye_landmarks[:, 0]))
    x_max = int(np.max(eye_landmarks[:, 0]))
    y_min = int(np.min(eye_landmarks[:, 1]))
    y_max = int(np.max(eye_landmarks[:, 1]))

    eye_width = x_max - x_min
    eye_height = y_max - y_min

    # Resize lash image to fit the eye
    lash_resized = cv2.resize(lash_image, (eye_width, eye_height))

    # Extract the alpha mask of the lash image
    alpha_s = lash_resized[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    # Overlay the lash image
    for c in range(0, 3):
        image[y_min:y_max, x_min:x_max, c] = (alpha_s * lash_resized[:, :, c] +
                                              alpha_l * image[y_min:y_max, x_min:x_max, c])

# Gradio Interface Function
def process_image(image_in, style_preference):
    # Convert PIL Image to OpenCV format
    image = cv2.cvtColor(np.array(image_in), cv2.COLOR_RGB2BGR)

    # Extract measurements and recommendations
    eye_measurements = extract_eye_measurements(image)
    if 'error' in eye_measurements:
        return "No faces detected", None

    lash_recommendations = recommend_lashes(eye_measurements, style_preference=style_preference)

    # Visualization
    for eye in eye_measurements.keys():
        data = eye_measurements[eye]
        rec = lash_recommendations[eye]

        # Draw landmarks
        draw_eye_landmarks(image, data['landmarks'])

        # Annotate measurements and eye shape
        annotate_eye_info(image, eye, data)

        # Annotate lash recommendations
        annotate_lash_info(image, eye, rec, data)

        # Overlay lash images
        lash_image_path = select_lash_image(rec)
        overlay_lash(image, data['landmarks'], lash_image_path)

    # Convert back to PIL Image
    image_out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return None, image_out

# Create Gradio Interface
style_options = ['Natural', 'Dramatic', 'Subtle']
iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.inputs.Image(type="pil", label="Upload an Image"),
        gr.inputs.Radio(style_options, label="Style Preference")
    ],
    outputs=[
        gr.outputs.Textbox(label="Message"),
        gr.outputs.Image(type="pil", label="Processed Image")
    ],
    title="Eyelash Recommendation",
    description="Upload an image, and get eyelash recommendations based on your eye shape."
)

if __name__ == '__main__':
    iface.launch()
