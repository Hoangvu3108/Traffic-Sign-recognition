import socket
import struct
import cv2
import numpy as np
import time
from tensorflow.lite.python.interpreter import Interpreter
import threading

# --- CONFIG ---
HOST = '192.168.1.123'  # IP của máy server để nhận ảnh và sensor
PORT = 8890  # Cổng để nhận từ C++ client (ảnh + sensor)

SIGNAL_IP = '192.168.1.137'  # IP của máy client chạy signal_receiver()
SIGNAL_PORT = 8888  # Cổng nhận tín hiệu điều khiển bên client

# Tham số nội tại của camera (dữ liệu hiệu chỉnh)
camera_matrix = np.array([[262.08953333143063, 0.0, 330.77574325128484],
                          [0.0, 263.57901348164575, 250.50298224489268],
                          [0.0, 0.0, 1.0]], dtype=np.float64)

# Hệ số méo ảnh (distortion coefficients)
dist_coeffs = np.array([-0.27166331922859776, 0.09924985737514846,
                        -0.0002707688044880526, 0.0006724194580262318,
                        -0.01935517123682299], dtype=np.float64)

model_path = r"C:\Users\Asus\Downloads\2024.2\datn\datn\custom_model_lite\detect.tflite"
label_path = r"C:\Users\Asus\Downloads\2024.2\datn\datn\custom_model_lite\labelmap.txt"
min_confidence = 0.5


class UndistortData:
    def __init__(self):
        self.roi = None
        self.map1 = None
        self.map2 = None


def setup_undistort(camera_matrix, dist_coeffs, image_size):
    data = UndistortData()
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, image_size, 0.4, image_size)

    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, image_size,
                                             cv2.CV_16SC2)

    data.roi = roi
    data.map1 = map1
    data.map2 = map2
    return data


def undistort_frame(frame, data):
    undistorted = cv2.remap(frame, data.map1, data.map2, cv2.INTER_LINEAR)
    undistorted = undistorted[data.roi[1]:data.roi[1] + data.roi[3], data.roi[0]:data.roi[0] + data.roi[2]]
    return undistorted


def recv_exact(sock, size):
    data = b''
    while len(data) < size:
        more = sock.recv(size - len(data))
        if not more:
            raise ConnectionError("Socket connection broken")
        data += more
    return data


def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def load_tflite_model(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def detect_objects(frame, interpreter, input_details, output_details, labels, min_confidence):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imH, imW, _ = frame.shape
    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]
    float_input = (input_details[0]['dtype'] == np.float32)
    image_resized = cv2.resize(image_rgb, (input_width, input_height))
    input_data = np.expand_dims(image_resized, axis=0)
    if float_input:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    signal = "none"
    distance = 0
    for i in range(len(scores)):
        if scores[i] > min_confidence and scores[i] <= 1.0:
            ymin = int(max(1, boxes[i][0] * imH))
            xmin = int(max(1, boxes[i][1] * imW))
            ymax = int(min(imH, boxes[i][2] * imH))
            xmax = int(min(imW, boxes[i][3] * imW))
            object_name = labels[int(classes[i])]
            box_width = xmax - xmin
            #distance = int((-0.4412) * box_width + 57.9706) # 57.9706
            distance = ((-0.4412) * box_width + 57.9706)  # Công thức tính khoảng cách
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"{object_name}: {int(scores[i] * 100)}% | {distance}cm"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            signal = classes[i]
    return distance, signal


def send_command_signal(msgID, angle):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as command_socket:
            command_socket.connect((SIGNAL_IP, SIGNAL_PORT))
            command = f"{msgID},{angle}"
            command_socket.sendall(command.encode())
            print(f"[PC] Sending command to Pi: {command}")
    except Exception as e:
        print(f"[PC] Failed to send command to Pi: {e}")


def send_simple_control_signal(signal_sock, command="10\n"):
    """Send a simple control command using the persistent signal_sock connection."""
    if signal_sock:
        try:
            signal_sock.sendall(command.encode())
            print(f"[CONTROL] Sent control signal: {command.strip()}")
        except Exception as e:
            print(f"[ERROR] Failed to send control signal: {e}")


def start_server():
    # TCP socket nhận ảnh và sensor
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"[SERVER] Listening for image/sensor data on {HOST}:{PORT}")
    client_socket, addr = server_socket.accept()
    print(f"[SERVER] Connected to data sender at {addr}")

    # TCP client gửi tín hiệu điều khiển tới client
    signal_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        signal_sock.connect((SIGNAL_IP, SIGNAL_PORT))
        print(f"[CONTROL] Connected to signal receiver at {SIGNAL_IP}:{SIGNAL_PORT}")
    except Exception as e:
        print(f"[ERROR] Could not connect to signal receiver: {e}")
        signal_sock = None

    prev_time = time.time()

    # Khởi tạo model và labels cho object detection
    interpreter, input_details, output_details = load_tflite_model(model_path)
    labels = load_labels(label_path)

    undistort_data = None
    image_size = None

    try:
        while True:
            # 1. Nhận kích thước ảnh (4 byte)
            try:
                img_size_data = recv_exact(client_socket, 4)
                img_size = struct.unpack('i', img_size_data)[0]
            except Exception as e:
                print(f"[ERROR] Failed to receive image size: {e}")
                break

            # 2. Nhận dữ liệu ảnh
            img_buf = recv_exact(client_socket, img_size)

            # 3. Giải mã ảnh
            image = cv2.imdecode(np.frombuffer(img_buf, dtype=np.uint8), cv2.IMREAD_COLOR)

            # Chỉ setup undistort map 1 lần duy nhất
            if undistort_data is None:
                image_size = (image.shape[1], image.shape[0])  # (width, height)
                undistort_data = setup_undistort(camera_matrix, dist_coeffs, image_size)

            # 4. Xử lý ảnh méo
            undistorted_image = undistort_frame(image, undistort_data)

            # 5. Nhận độ dài dữ liệu cảm biến (4 byte)
            try:
                txt_len_data = recv_exact(client_socket, 4)
                txt_len = struct.unpack('i', txt_len_data)[0]
            except Exception as e:
                print(f"[ERROR] Failed to receive sensor text length: {e}")
                break

            # 6. Nhận dữ liệu cảm biến (text)
            text_buf = recv_exact(client_socket, txt_len)
            sensor_data = text_buf.decode()
            print(f"[RECEIVED SENSOR DATA] {sensor_data}")

            # 7. Tính FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # 8. Phát hiện đối tượng và gửi lệnh điều khiển
            distance_output, signal2 = detect_objects(
                undistorted_image, interpreter, input_details, output_details, labels, min_confidence)

            # 9. Vẽ FPS và hiển thị ảnh
            if undistorted_image is not None:
                cv2.putText(undistorted_image, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Received Image', undistorted_image)
                if cv2.waitKey(1) == ord('q'):
                    break

            # --- Control signal block: chỉ gửi qua signal_sock ---
            if signal_sock:
                NORMAL_SPEED = 10
                STOP = 0
                SLOW = 5
                command = None
                # Logic điều khiển tương ứng với từng loại biển báo
                if signal2 == "none":
                    command = f"{NORMAL_SPEED}\n"
                elif signal2 == 8.0:  # Biển báo dừng (STOP sign)
                    if distance_output < 50:
                        command = f"{STOP}\n"  # STOP, không gửi lại 10
                elif signal2 == 0.0:
                    if distance_output < -30:
                        command = f"{STOP}\n"
                elif signal2 == 2.0:  # Biển báo có người đi bộ
                    if  distance_output < 40:
                        command = f"{SLOW}\n"
                        try:
                            signal_sock.sendall(command.encode())
                            print(f"[CONTROL] Sent control signal: {command.strip()}")
                        except Exception as e:
                            print(f"[ERROR] Failed to send control signal: {e}")
                            break
                        # time.sleep(0.1)
                        # command = f"{NORMAL_SPEED}\n"
                        # try:
                        #     signal_sock.sendall(command.encode())
                        #     print(f"[CONTROL] Sent control signal: {command.strip()}")
                        # except Exception as e:
                        #     print(f"[ERROR] Failed to send control signal: {e}")
                        #     break
                        # time.sleep(0.1)
                        continue
                elif signal2 == 6.0:  # Đèn vàng
                    command = f"{SLOW}\n"
                    try:
                        signal_sock.sendall(command.encode())
                        print(f"[CONTROL] Sent control signal: {command.strip()}")
                    except Exception as e:
                        print(f"[ERROR] Failed to send control signal: {e}")
                        break
                    time.sleep(0.1)
                    continue
                elif signal2 == 5.0:  # Đèn đỏ
                    if  distance_output < 50:
                        command = f"{STOP}\n"
                        try:
                            signal_sock.sendall(command.encode())
                            print(f"[CONTROL] Sent control signal: {command.strip()}")
                        except Exception as e:
                            print(f"[ERROR] Failed to send control signal: {e}")
                            break
                        time.sleep(0.01)
                        # command = f"{NORMAL_SPEED}\n"
                        # try:
                        #     signal_sock.sendall(command.encode())
                        #     print(f"[CONTROL] Sent control signal: {command.strip()}")
                        # except Exception as e:
                        #     print(f"[ERROR] Failed to send control signal: {e}")
                        #     break
                        # time.sleep(0.02)
                        continue
                elif signal2 == 4.0:  # Đèn xanh
                    command = f"{NORMAL_SPEED}\n"
                # Gửi lệnh nếu có
                if command is not None:
                    try:
                        signal_sock.sendall(command.encode())
                        print(f"[CONTROL] Sent control signal: {command.strip()}")
                    except Exception as e:
                        print(f"[ERROR] Failed to send control signal: {e}")
                        break
# ...existing code...
    
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        client_socket.close()
        server_socket.close()
        if signal_sock:
            signal_sock.close()
        cv2.destroyAllWindows()
        print("[SERVER] Closed all connections.")


if __name__ == '__main__':
    start_server()
