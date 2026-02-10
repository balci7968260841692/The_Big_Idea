import cv2
import mediapipe as mp
import pyautogui
import random
import util
import queue
import threading
from pynput.mouse import Button, Controller
from pynput import keyboard as pynput_keyboard
import Voice_keyboard

mouse = Controller()

screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)


def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None, None


def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y / 2 * screen_height)
        pyautogui.moveTo(x, y)


def is_left_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            thumb_index_dist > 50
    )


def is_right_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90  and
            thumb_index_dist > 50
    )


def is_double_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist > 50
    )


def is_screenshot(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist < 50
    )


def detect_gesture(frame, landmark_list, processed):
    if len(landmark_list) >= 21:

        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])

        if util.get_distance([landmark_list[4], landmark_list[5]]) < 50  and util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            move_mouse(index_finger_tip)
        elif is_left_click(landmark_list,  thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif is_screenshot(landmark_list,thumb_index_dist ):
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


def execute_voice_message(msg):
    """Execute voice commands"""
    msg_type = msg.get("type")
    
    if msg_type == "TYPE":
        text = msg.get("text", "")
        pyautogui.write(text, interval=0.05)
    
    elif msg_type == "KEY":
        key = msg.get("key")
        pyautogui.press(key)
    
    elif msg_type == "PYNPUT_KEY":
        # For media keys that pyautogui doesn't handle well
        key_name = msg.get("key")
        key_obj = getattr(pynput_keyboard.Key, key_name, None)
        if key_obj:
            kb = pynput_keyboard.Controller()
            kb.press(key_obj)
            kb.release(key_obj)
    
    elif msg_type == "HOTKEY":
        keys = msg.get("keys", [])
        pyautogui.hotkey(*keys)
    
    elif msg_type == "SCROLL":
        direction = msg.get("dir")
        if direction == "down":
            pyautogui.scroll(-3)
        elif direction == "up":
            pyautogui.scroll(3)
    
    elif msg_type == "MOUSE":
        button = msg.get("button")
        if button == "left":
            mouse.press(Button.left)
            mouse.release(Button.left)
        elif button == "right":
            mouse.press(Button.right)
            mouse.release(Button.right)
    
    elif msg_type == "SEARCH":
        text = msg.get("text", "")
        pyautogui.write(text, interval=0.05)
        pyautogui.press("enter")


def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    # Setup voice control
    voice_queue = queue.Queue()
    stop_event = threading.Event()
    voice_thread = threading.Thread(
        target=Voice_keyboard.voice_worker,
        args=(voice_queue, stop_event),
        daemon=True
    )
    voice_thread.start()
    
    print("TRACKER Started!")
    print("Hand tracking: Active")
    print("Voice control: Hold F9 to speak")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            detect_gesture(frame, landmark_list, processed)
            
            # Process voice commands
            try:
                while True:
                    msg = voice_queue.get_nowait()
                    print(f"Voice command: {msg}")
                    execute_voice_message(msg)
            except queue.Empty:
                pass

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
