import logging
import time
import random
import requests
import os
import re
from typing import List, Dict
import shutil
import json
import base64
from pathlib import Path


from .constants import classes, landing_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation
# from ultralytics import YOLO

gt_translation_file_path = "ground_truth.json"
gt_directory = os.path.dirname(gt_translation_file_path)
if gt_directory:  # Only create if there's a directory path
    os.makedirs(gt_directory, exist_ok=True)
orb_slam3_url = "http://localhost"
orb_slam3_port = 5000     

class ObjectDetectionModel:
    # Base class for team models

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        self.shared_volume_path = "../shared"
        # self.model = YOLO("best.pt")
        # device == "auto"
        # if device == 'auto':
        #     self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # else:
        #     self.device = device
        # Modelinizi bu kısımda init edebilirsiniz.
        # self.model = get_keras_model() # Örnektir!

    @staticmethod
    def download_image(img_url, images_folder, images_files, retries=3, initial_wait_time=0.1):
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        # Indirmek istedigimiz frame frames.json dosyasinda mevcut mu kontrol edelim
        image_name = img_url.split("/")[-1]
        # Eger indirecegimiz frame'i daha once indirmediysek indirme islemine gecelim
        if image_name not in images_files:
            for attempt in range(retries):
                    try:
                        response = requests.get(img_url, timeout=60)
                        response.raise_for_status()
                        
                        img_bytes = response.content
                        with open(images_folder + image_name, 'wb') as img_file:
                            img_file.write(img_bytes)

                        t2 = time.perf_counter()
                        logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')
                        return

                    except requests.exceptions.RequestException as e:
                        logging.error(f"Download failed for {img_url} on attempt {attempt + 1}: {e}")
                        logging.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        wait_time *= 2

            logging.error(f"Failed to download image from {img_url} after {retries} attempts.")
        # Eger indirecegimiz frame'i daha once indirdiysek indirme yapmadan devam edebiliriz
        else:
            logging.info(f'{image_name} already exists in {images_folder}, skipping download.')

    def process(self, prediction,evaluation_server_url,health_status, images_folder ,images_files):
        # Yarışmacılar resim indirme, pre ve post process vb işlemlerini burada gerçekleştirebilir.
        # Download image (Ornek)
        self.download_image(evaluation_server_url + "media" + prediction.image_url, images_folder, images_files)
        # Örnek: Burada OpenCV gibi bir tool ile preprocessing işlemi yapılabilir. (Tercihe Bağlı)

        ### CREATE A COPY OF THE FILE
        cur_image_file_path = images_folder + prediction.image_url.split("/")[-1]  # İndirilen resmin tam yolu
        copied_cur_image_file_path = os.path.join(self.shared_volume_path, prediction.image_url.split("/")[-1])
        if not os.path.isfile(copied_cur_image_file_path):
            try:
                logging.info(f"TRYING COPY: {cur_image_file_path} to \n{copied_cur_image_file_path}")
                shutil.copy(cur_image_file_path, copied_cur_image_file_path)
                logging.info(f"Copied {cur_image_file_path} → {copied_cur_image_file_path}")
            except Exception as e:
                logging.error(f"Failed to copy file: {e}")
        else:
            logging.info(f"{copied_cur_image_file_path} already exists, skipping copy.")


        # Nesne tespiti ve pozisyon kestirim modelinin bulunduğu fonksiyonun (self.detect() ) çağırılması burada olmalıdır.
        frame_results = self.detect(prediction, health_status, copied_cur_image_file_path)

        ### CREATE A GROUNDTRUTH JSON FILE
        new_frame_key = os.path.basename(prediction.image_url)  # e.g. "frame_000400.jpg"
        if health_status == 0:
            vals = frame_results
        else:
            vals = prediction

        new_frame_value = {
            "translation_x": vals.gt_translation_x,
            "translation_y": vals.gt_translation_y,
            "translation_z": vals.gt_translation_z
        }


        # 2) load or init
        try:
            with open(gt_translation_file_path, 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}

        # 3) update using the key
        existing_data[new_frame_key] = new_frame_value

        # 4) write back
        with open(gt_translation_file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)

        ### If frame is "frame_001796.jpg", calculate scale and rotation
        if (os.path.basename(prediction.image_url)) == "frame_001796.jpg":
            copied_gt_translation_file_path = os.path.join(self.shared_volume_path, gt_translation_file_path)
            shutil.copy(gt_translation_file_path, copied_gt_translation_file_path)
            self.calculate_scale_and_rotation(initial_frames_file_path_list=self.get_previous_frames(copied_cur_image_file_path, 450, 4), gt_file_path=copied_gt_translation_file_path, session_name=prediction.video_name)

        # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.
        return frame_results
    
    def boxes_intersect(self, box1, box2):
            return not (
                box1["bottom_right_x"] < box2["top_left_x"] or
                box1["top_left_x"] > box2["bottom_right_x"] or
                box1["bottom_right_y"] < box2["top_left_y"] or
                box1["top_left_y"] > box2["bottom_right_y"]
            )


    def detect(self, prediction, health_status, copied_cur_image_file_path):
        #### BURAYA
        
        # results = self.model("best.pt")[0]

        for i in range(1, 3):
            cls = classes["UAP"],  # Tahmin edilen nesnenin sınıfı classes sözlüğü kullanılarak atanmalıdır.
            landing_status = landing_statuses["Inilebilir"]  # Tahmin edilen nesnenin inilebilir durumu landing_statuses sözlüğü kullanılarak atanmalıdır.
            top_left_x = 12 * i  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
            top_left_y = 12 * i  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
            bottom_right_x = 12 * i  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
            bottom_right_y = 12 * i  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.

            # Modelin tespit ettiği herbir nesne için bir DetectedObject sınıfına ait nesne oluşturularak
            # tahmin modelinin sonuçları parametre olarak verilmelidir.
            d_obj = DetectedObject(cls,
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y
                                        )

            # Modelin tahmin ettiği her nesne prediction nesnesi içerisinde bulunan detected_objects listesine eklenmelidir.
            prediction.add_detected_object(d_obj)


        if health_status == '0':
            try:
                orb_slam_result_for_current_frame = self.connect_to_orb_slam(copied_cur_image_file_path)
                cur_frame_local_translation = orb_slam_result_for_current_frame['translation'][0]

                # check prev frame's translation data:
                prev_frame_file_path = self.decrement_frame_path(copied_cur_image_file_path, dec=4)
                is_prev_processed, prev_frame_result = self.check_previous_frames_translation_data(prev_frame_file_path)

                cur_global_translation = {}
                if is_prev_processed:
                    # compute global translation
                    prev_local_translation = prev_frame_result["translation"][0]
                    cur_global_translation = self.find_global_translation(
                        prediction,                          
                        prev_local_translation, 
                        self.read_gt_and_get_global_translation(gt_file_path=gt_translation_file_path,frame_name=os.path.basename(prev_frame_file_path)), 
                        cur_frame_local_translation,
                        prediction.video_name
                    )
                else:
                    orb_slam_prev_frame_result = self.connect_to_orb_slam(prev_frame_file_path)
                    prev_frame_local_translation = orb_slam_prev_frame_result['translation'][0]
                    # first process the previous frame
                    # then compute global translation
                    cur_global_translation = self.find_global_translation(
                        prediction,                          
                        prev_frame_local_translation, 
                        self.read_gt_and_get_global_translation(gt_file_path=gt_translation_file_path,frame_name=os.path.basename(prev_frame_file_path)), 
                        cur_frame_local_translation,
                        prediction.video_name
                    )

                # Takimlar buraya kendi gelistirdikleri algoritmalarin sonuclarini entegre edebilirler.
                pred_translation_x = cur_global_translation['translation']['x']
                pred_translation_y = cur_global_translation['translation']['y']
                pred_translation_z = cur_global_translation['translation']['z']
                
            except Exception as e:
                # Handle ORB-SLAM error by getting previous frame's global translation
                print(f"Error in ORB-SLAM processing: {e}")
                
                # Get previous frame's global translation as fallback
                prev_frame_file_path = self.decrement_frame_path(copied_cur_image_file_path, dec=4)
                prev_frame_global_translation = self.read_gt_and_get_global_translation(
                    gt_file_path=gt_translation_file_path,
                    frame_name=os.path.basename(prev_frame_file_path)
                )
                
                # Use previous frame's global translation as current prediction
                pred_translation_x = prev_frame_global_translation[0]
                pred_translation_y = prev_frame_global_translation[1]
                pred_translation_z = prev_frame_global_translation[2]

            # pred_translation_x = random.randint(1, 10) # Ornek olmasi icin rastgele degerler atanmistir takimlar kendi sonuclarini kullanmalidirlar.
            # pred_translation_y = random.randint(1, 10) # Ornek olmasi icin rastgele degerler atanmistir takimlar kendi sonuclarini kullanmalidirlar.
            # pred_translation_z = random.randint(1, 10)               
        else:
            pred_translation_x = prediction.gt_translation_x # String
            pred_translation_y = prediction.gt_translation_y
            pred_translation_z = prediction.gt_translation_z

        # Translation icin hesaplanilan degerleri sunucuya gondermek icin ilgili objeye dolduralim.
        trans_obj = DetectedTranslation(pred_translation_x, pred_translation_y, pred_translation_z)
        prediction.add_translation_object(trans_obj)

        return prediction
    
    def read_gt_and_get_global_translation(self, gt_file_path: str, frame_name: str):
        """
        Given:
          - gt_file_path: path to your ground_truth.json
          - frame_name:    e.g. "frame_000400.jpg"
        Returns:
          [x, y, z] for that frame, or raises a KeyError if missing.
        """
        # 1) Load the JSON
        try:
            with open(gt_file_path, 'r') as f:
                gt_dict = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"GT file not found: {gt_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"GT JSON is invalid: {e}")

        # 2) Look up the frame
        if frame_name not in gt_dict:
            raise KeyError(f"No GT translation found for frame: {frame_name}")

        entry = gt_dict[frame_name]
        
        # 3) Validate and extract
        try:
            x = float(entry['translation_x'])
            y = float(entry['translation_y'])
            z = float(entry['translation_z'])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Invalid GT data for frame {frame_name}: {entry}")

        return [x, y, z]
    
    def read_image_as_base64(image_path: str) -> str:
        """
        Reads an image file and returns it as a base64-encoded string.
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')

    def connect_to_orb_slam(self, cur_frame_file_path: str, slice: int = 5):
        """
        Sends the current frame plus a slice of previous frames
        to the ORB-SLAM3 /process_frame endpoint in the correct JSON format.
        Returns the parsed JSON on success or a dict with 'error' on failure.
        """
        try:
            # 1) build list of previous frame-paths
            previous_paths = self.get_previous_frames(
                cur_frame_file_path,
                count=slice,
                step=4
            )
        except Exception as e:
            return {'error': f'Failed to get previous frames: {e}'}

        try:
            # 2) Read current frame data
            cur_frame_data = self.read_image_as_base64(cur_frame_file_path)
            cur_frame_filename = os.path.basename(cur_frame_file_path)
            
            # 3) Read previous frames data
            previous_frames_data = []
            for frame_path in previous_paths:
                frame_data = self.read_image_as_base64(frame_path)
                frame_filename = os.path.basename(frame_path)
                previous_frames_data.append({
                    "image_data": frame_data,
                    "filename": frame_filename
                })
            
            # 4) prepare payload
            data = {
                "new_frame_data": {
                    "image_data": cur_frame_data,
                    "filename": cur_frame_filename
                },
                "new_frame_name": os.path.basename(cur_frame_file_path).rsplit('.', 1)[0],
                "previous_frames": previous_frames_data
            }
        except Exception as e:
            return {'error': f'Failed to build request payload: {e}'}

        try:
            endpoint = "process_frame"
            # 5) POST to your endpoint
            orb_slam_url = f"{orb_slam3_url}:{orb_slam3_port}/{endpoint}"
            headers = {"Content-Type": "application/json"}
            resp = requests.post(orb_slam_url, headers=headers, json=data, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            return {'error': f'HTTP request failed: {e}'}
        except Exception as e:
            return {'error': f'Unexpected error during request: {e}'}

        try:
            # 6) return parsed JSON
            return resp.json()
        except ValueError as e:
            return {'error': f'Failed to parse JSON response: {e}', 'status_code': resp.status_code, 'text': resp.text}

    def get_previous_frames(self, cur_frame_file_path: str, count: int = 5, step: int = 4):
        """
        Updated version that returns just the file paths (no more image_path wrapper).
        """
        try:
            # Extract directory and current frame info
            frame_dir = Path(cur_frame_file_path).parent
            frame_name = Path(cur_frame_file_path).stem
            frame_ext = Path(cur_frame_file_path).suffix
            
            # Extract frame number (assuming format like "frame_000123")
            frame_num_str = frame_name.split('_')[-1]
            current_frame_num = int(frame_num_str)
            
            previous_frames = []
            
            # Get previous frames with the specified step
            for i in range(1, count + 1):
                prev_frame_num = current_frame_num - (i * step)
                if prev_frame_num < 0:
                    break
                    
                # Format the frame number with leading zeros
                prev_frame_name = f"frame_{prev_frame_num:06d}"
                prev_frame_path = frame_dir / f"{prev_frame_name}{frame_ext}"
                
                if prev_frame_path.exists():
                    previous_frames.append(str(prev_frame_path))
            
            return previous_frames
            
        except Exception as e:
            raise Exception(f"Error getting previous frames: {e}")
        
    def decrement_frame_path(self, path: str, dec: int = 4) -> str:
        # Split into directory and filename
        dir_name, filename = os.path.split(path)
        
        # Regex to grab prefix, digits, and extension
        m = re.match(r'^(.*?)(\d+)(\.\w+)$', filename)
        if not m:
            raise ValueError(f"Filename '{filename}' doesn't match the expected pattern")
        prefix, num_str, ext = m.groups()
        
        # Compute new number and re-zero-pad
        width = len(num_str)
        new_idx = int(num_str) - dec
        if new_idx < 0:
            raise ValueError("Resulting frame index would be negative")
        new_num_str = str(new_idx).zfill(width)
        
        # Rebuild filename and full path
        new_filename = f"{prefix}{new_num_str}{ext}"
        return os.path.join(dir_name, new_filename)
    
    def find_global_translation(self, prediction, prev_local_translation, prev_global_translation, cur_local_translation, session):
        endpoint = "calculate_vehicle_position"
        # 3) POST to your endpoint
        # 1️⃣ Build your JSON payload:
        data = {
            "prev_global_translation": prev_local_translation,
            "prev_local_translation": prev_global_translation,
            "new_local_translation": cur_local_translation,
            
            "session_name": prediction.video_name
        }
        try:
            # 4) POST to your endpoint
            orb_slam_url = f"{orb_slam3_url}:{orb_slam3_port}/{endpoint}"
            headers = {"Content-Type": "application/json"}
            resp = requests.post(orb_slam_url, headers=headers, json=data, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            return {'error': f'HTTP request failed: {e}'}
        except Exception as e:
            return {'error': f'Unexpected error during request: {e}'}

    def calculate_scale_and_rotation(self, initial_frames_file_path_list, gt_file_path, session_name):
        endpoint = "process_initial_frames"

        # 1️⃣ Build your JSON payload:
        data = {
            "frames": [
                {fp} 
                for fp in initial_frames_file_path_list
            ],
            "gt_file_path": gt_file_path,
            "session_name": session_name
        }
        try:
            # 4) POST to your endpoint
            orb_slam_url = f"{orb_slam3_url}:{orb_slam3_port}/{endpoint}"
            headers = {"Content-Type": "application/json"}
            resp = requests.post(orb_slam_url, headers=headers, json=data, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            return {'error': f'HTTP request failed: {e}'}
        except Exception as e:
            return {'error': f'Unexpected error during request: {e}'}
