# **Pedestrian Detection Application**
This project designed to identify pedestrians in images and videos using advanced computer vision techniques. The application leverages the power of deep learning models, integrating multiple tools and technologies to deliver accurate and efficient pedestrian detection. It also provides a user-friendly interface .

## **Features**
- **Pedestrian Detection in Images:**  Detect pedestrians in static images, displaying bounding boxes around detected individuals. <br>

- **Pedestrian Detection in Videos:** Process video files to identify and track pedestrians frame by frame.<br>

- **Visualization of Results:**  Visualize detection results directly on the interface, with bounding boxes . <br>

- **Robust Performance:**  Uses the YOLO (You Only Look Once) model, fine-tuned with custom data for improved pedestrian detection.<br>


## **Technologies Used**

- **Roboflow:**   Used to prepare and annotate the pedestrian dataset for training the model.  <br>
- **YOLOv8 (Ultralytics):**   A state-of-the-art object detection model trained on the pedestrian dataset to achieve high accuracy in detection tasks.<br>
- **Flask:**   A lightweight Python framework used to develop the backend of the application.<br>
- **HTML/CSS/JavaScript:**  For designing the frontend and user interface of the application.<br>
- **OpenCV:**  A computer vision library used for image and video processing, as well as real-time detection.

## **Model Training**
The pedestrian detection model is trained using the YOLOv8 architecture, leveraging custom training data provided by the Roboflow dataset. The training process includes multiple optimization techniques to improve the model's performance, such as:

- **Optimizer:** NAdam optimizer with weight decay and momentum adjustments.
- **Learning Rate Schedule:** Cosine learning rate scheduling to improve convergence during training.
- **Data Augmentation:** Close mosaic augmentation and label smoothing are used to improve generalization.
- **Dropout:** A dropout rate of 0.5 is used to prevent overfitting.

### **Using `best.pt` Weights**
After training, the model generates a file called `best.pt`, which contains the best-performing weights based on the evaluation metrics. To use these weights in the application:

1. **Train the model:** Run the training script to fine-tune the YOLOv8 model using your custom dataset.
2. **Obtain `best.pt`:** After the training process completes, the best weights are saved as `best.pt`.
3. **Load `best.pt` for Inference:** 
    - Load the model using the saved weights (`best.pt`) to perform detection on images or videos:
    ```python
    model = YOLO('path_to_best.pt')
    results = model.predict('image_or_video_path')
    ```

 

