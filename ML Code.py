import paramiko
import yolov5

# Raspberry Pi SSH connection settings
hostname = '192.168.93.64'
port = 22
username = 'pi'
password = 'raspberry'

# Remote directory and file name
remote_directory = '/home/pi/Desktop/GarbageDetection/PhotoOutput'
remote_filename = 'image.jpg'

# Local directory where the image will be saved on your PC
local_directory = 'C:\\Users\\mhdat\\Desktop\\EPICS\\Epics\\Input'
local_filename = 'image.jpg'

# Establish SSH connection to Raspberry Pi
try:
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, port, username, password)

    # Create SCP client
    scp_client = ssh_client.open_sftp()

    # Download the image file
    remote_file_path = f"{remote_directory}/{remote_filename}"
    local_file_path = f"{local_directory}\\{local_filename}"
    scp_client.get(remote_file_path, local_file_path)

    print(f"Image downloaded from {remote_file_path} and saved to {local_file_path}")

finally:
    # Close SCP and SSH clients
    if 'scp_client' in locals():
        scp_client.close()
    if 'ssh_client' in locals():
        ssh_client.close()

# load model
model = yolov5.load('keremberke/yolov5m-garbage')
  
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# set image
img = 'C:\\Users\\mhdat\\Desktop\\EPICS\\Epics\\Input\\image.jpg'

# perform inference
results = model(img, size=640)

# inference with test time augmentation
results = model(img, augment=True)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4] # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# Mapping of category indices to category names
category_names = {
    0: 'Biodegradable',
    1: 'Carboard',
    2: 'Glass',
    3: 'Metal',
    4: 'Paper',
    5: 'Plastic'
}

# Initialize an empty dictionary to store the mapping
class_names = {}

# Iterate through predictions and assign category names dynamically
for box, category in zip(boxes, categories):
    class_index = int(category)
    class_names[class_index] = category_names.get(class_index, f"Class {class_index}")

# Iterate through predictions and print details with dynamically assigned category names
for box, score, category in zip(boxes, scores, categories):
    x1, y1, x2, y2 = box
    class_index = int(category)
    class_name = class_names.get(class_index, 'Unknown')
    print(class_name)

output_text = class_name
output_file_path = 'C:\\Users\\mhdat\\Desktop\\EPICS\\Epics\\Output\\prediction_output.txt'

# Write the output text to the specified file
with open(output_file_path, 'w') as file:
    file.write(output_text)



# Local directory containing the text file in Google Colab
local_directory = 'C:\\Users\\mhdat\\Desktop\\EPICS\\Epics\\Output'
local_filename = 'prediction_output.txt'

# Remote directory where the text file will be saved on Raspberry Pi
remote_directory = '/home/pi/Desktop/GarbageDetection/TextInput'

# Establish SSH connection to Raspberry Pi
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname, port, username, password)

# Create SCP client
scp_client = ssh_client.open_sftp()

# Upload the text file
local_file_path = f"{local_directory}/{local_filename}"
remote_file_path = f"{remote_directory}/{local_filename}"
scp_client.put(local_file_path, remote_file_path)