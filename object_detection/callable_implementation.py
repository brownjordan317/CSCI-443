import csv
import os
import os
import torch
from torchvision import models, transforms
from datetime import datetime, timedelta
import numpy as np
import cv2

def write_csv(data, filename):
    # Define the correct order of columns
    headers = ["Time", 
               "Day of the week",
               "CarCount",
               "BikeCount",
               "BusCount",
               "TruckCount",
               "Total"]
    
    # Check if the file already exists
    file_exists = os.path.exists(filename)
    
    # Open the file in append mode
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file doesn't exist
        if not file_exists:
            writer.writerow(headers)
        
        # Write the row in the correct order
        writer.writerow([data.get(header, "") for header in headers])

def load_model():
    # Load a pre-trained Faster R-CNN model with the option to use GPU
    model = models.detection.fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Correct COCO categories for vehicles
    vehicle_classes = {
        1: "Person",       # COCO ID for Person
        # 17: "Cat",          # COCO ID for Cat
        # 18: "Dog",          # COCO ID for Dog
        3: "Car",          # COCO ID for Car
        4: "Bike",         # COCO ID for Bike
        6: "Bus",          # COCO ID for Bus
        8: "Truck"         # COCO ID for Truck
    }    
    
    return model, vehicle_classes, device

def load_tensor_img(image_path, device):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB as the model expects

    # Transformation to convert image to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)  # Add batch dimension and move to device

    return img, img_tensor

# Function to process a single image
def process_image(model, vehicle_classes, img, img_tensor, output_path, output_filename, threshold=0.75):
    # Run the model on the image
    with torch.no_grad():
        prediction = model(img_tensor)

    # Extract prediction details
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    # Filter predictions based on the confidence threshold
    mask = scores > threshold
    valid_boxes = boxes[mask]
    valid_labels = labels[mask]
    valid_scores = scores[mask]

    counts = {
        "PersonCount": 0,
    #           "CatCount": 0,
    #           "DogCount": 0,
              "CarCount": 0, 
              "BikeCount": 0, 
              "BusCount": 0, 
              "TruckCount": 0}

    for i, box in enumerate(valid_boxes):
        xmin, ymin, xmax, ymax = map(int, box.tolist())
        label = valid_labels[i].item()
        score = valid_scores[i].item()

        # Check if the detected object is a vehicle
        if label in vehicle_classes:
            category = vehicle_classes[label]
            counts[f"{category}Count"] += 1
            
            # Draw rectangle and label on the image
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)  # Red box
            cv2.putText(img, f"{category}: {score:.2f}", (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1)  # Yellow text

    # Save the image with bounding boxes
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, os.path.basename(output_filename))
    
    cv2.imwrite(output_file, img)

    # Add total count to the dictionary
    counts["Total"] = sum(counts.values())
    
    return counts, img
    
def get_dt_info():
    # get current time
    now = datetime.now()
    modified_time_dt_str = now.strftime("%H%M%S")
    # get day of the week
    day_of_week = now.strftime("%A")
    return modified_time_dt_str, day_of_week
    
        

if __name__ == "__main__":
    image_path = 'object_detection/download.jpg'  # Replace with your directory path
    output_path = 'test_images'
    # create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model, vehicle_classes, device = load_model()
    img, img_tensor = load_tensor_img(image_path, device)
    data, out_image = process_image(model, vehicle_classes, img, img_tensor, output_path, output_filename="test.jpg")
    time, day = get_dt_info()
    data["Time"] = time
    data["Day of the week"] = day
    write_csv(data, "test.csv")
    
    # send data to model
    # returns high, nuetral, low
    

