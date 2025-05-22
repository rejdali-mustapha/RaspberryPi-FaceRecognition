import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import time
import os
from picamera2 import Picamera2
import threading
import RPi.GPIO as GPIO

# Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# GPIO Pins
LED_PIN = 23          # Green LED
LED_PIN_RED =27      # Blinking red LED
BUZZER_PIN = 18       # Buzzer
SERVO_PIN = 17        # Servo control pin

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(LED_PIN_RED, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# PWM Setup
buzzer_pwm = GPIO.PWM(BUZZER_PIN, 1000)  # 1kHz for buzzer
servo_pwm = GPIO.PWM(SERVO_PIN, 50)      # 50Hz for servo
servo_pwm.start(0)
buzzer_pwm_active = False

# Global shared state
frame_buffer = None
latest_prediction = None
latest_confidence = 0
latest_top_predictions = []
running = True
blinking = False
servo_activated = False
frame_lock = threading.Lock()
result_lock = threading.Lock()


# Load and run models
def load_tflite_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def get_tflite_model_details(interpreter):
    return interpreter.get_input_details(), interpreter.get_output_details()


def predict_with_tflite(interpreter, input_data):
    input_details, output_details = get_tflite_model_details(interpreter)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


def preprocess_frame(frame, image_size):
    resized = cv2.resize(frame, image_size, interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(resized, axis=0).astype(np.float32) / 255.0


# Your servo function
def set_angle(angle):
    duty = 2 + (angle / 18)
    GPIO.output(SERVO_PIN, True)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(SERVO_PIN, False)
    servo_pwm.ChangeDutyCycle(0)


def led_blink_thread():
    while running:
        if blinking:
            GPIO.output(LED_PIN_RED, GPIO.HIGH)
            time.sleep(0.2)
            GPIO.output(LED_PIN_RED, GPIO.LOW)
            time.sleep(0.2)
        else:
            time.sleep(0.1)


def prediction_thread(model_interpreter, extractor_interpreter, image_size, class_names):
    global frame_buffer, latest_prediction, latest_confidence, latest_top_predictions
    global running, blinking, buzzer_pwm_active, servo_activated

    last_predictions = []
    buffer_size = 3

    while running:
        with frame_lock:
            if frame_buffer is None:
                time.sleep(0.01)
                continue
            current_frame = frame_buffer.copy()

        try:
            processed = preprocess_frame(current_frame, image_size)
            features = predict_with_tflite(extractor_interpreter, processed)
            features_flat = features.reshape(1, -1).astype(np.float32)
            prediction = predict_with_tflite(model_interpreter, features_flat)

            last_predictions.append(prediction[0])
            if len(last_predictions) > buffer_size:
                last_predictions.pop(0)

            avg_prediction = np.mean(last_predictions, axis=0)
            top_index = np.argmax(avg_prediction)
            top_preds = [(class_names[idx], avg_prediction[idx] * 100) for idx in np.argsort(avg_prediction)[-3:][::-1]]

            with result_lock:
                latest_prediction = class_names[top_index]
                latest_confidence = avg_prediction[top_index] * 100
                latest_top_predictions = top_preds

                if latest_prediction.lower() in ("rhandor", "mustapha", "sadik"):
                    blinking = False
                    GPIO.output(LED_PIN, GPIO.HIGH)

                    if not servo_activated:
                        set_angle(180)
                        servo_activated = True
                    if buzzer_pwm_active:
                        buzzer_pwm.stop()
                        buzzer_pwm_active = False
                else:
                    blinking = True
                    GPIO.output(LED_PIN, GPIO.LOW)
                    servo_activated = False
                    if not buzzer_pwm_active:
                        buzzer_pwm.start(50)
                        buzzer_pwm_active = True

        except Exception as e:
            print(f"Prediction error: {e}")

        time.sleep(0.05)


def test_with_picamera_tflite(model_path, feature_extractor_path, image_size, class_names):
    global frame_buffer, latest_prediction, latest_confidence, latest_top_predictions
    global running, blinking, buzzer_pwm_active

    try:
        model_interpreter = load_tflite_model(model_path)
        extractor_interpreter = load_tflite_model(feature_extractor_path)
    except Exception as e:
        print(f"Model loading error: {e}")
        return

    pred_thread = threading.Thread(
        target=prediction_thread,
        args=(model_interpreter, extractor_interpreter, image_size, class_names),
        daemon=True
    )
    blink_thread = threading.Thread(target=led_blink_thread, daemon=True)

    pred_thread.start()
    blink_thread.start()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}, buffer_count=4
    )
    picam2.configure(config)
    picam2.start()

    window_name = "Raspberry Pi - Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()
    fps = 0

    try:
        while True:
            frame = picam2.capture_array()

            with frame_lock:
                frame_buffer = frame

            frame_count += 1
            if time.time() - start_time >= 1.0:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()
                print(f"Camera FPS: {fps:.2f}")

            with result_lock:
                pred = latest_prediction
                conf = latest_confidence
                top_preds = latest_top_predictions.copy()

            disp = frame.copy()
            cv2.putText(disp, f"{pred}: {conf:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(disp, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            y_offset = 120
            for label, prob in top_preds:
                cv2.putText(disp, f"{label}: {prob:.1f}%", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 40

            cv2.imshow(window_name, disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        running = False
        pred_thread.join(timeout=1.0)
        blink_thread.join(timeout=1.0)
        picam2.stop()
        cv2.destroyAllWindows()
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.output(LED_PIN_RED, GPIO.LOW)
        if buzzer_pwm_active:
            buzzer_pwm.stop()
        servo_pwm.stop()
        GPIO.cleanup()
        print("Exited cleanly.")


def main():
    model_path = "models/face_recognition_model.tflite"
    extractor_path = "models/feature_extractor.tflite"
    class_names_path = "models/class_names.txt"
    image_size = (128, 128)

    try:
        with open(class_names_path, "r") as f:
            class_names = [line.strip() for line in f]
    except Exception as e:
        print(f"Error loading class names: {e}")
        return

    test_with_picamera_tflite(model_path, extractor_path, image_size, class_names)


if __name__ == "__main__":
    main()
