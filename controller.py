import cv2
import mediapipe as mp
import pyautogui
import keyboard
import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import math
import time

# ===================== CONFIG =====================
MODEL_PATH = r"D:\8th sem project alteration\Voice model"
SAMPLE_RATE = 16000

pyautogui.FAILSAFE = False

SMOOTHING_ALPHA = 0.18

SENSITIVITY = {
    "low":    2.0,
    "medium": 4.5,
    "high":   8.0
}

SCALING_FACTOR = SENSITIVITY["low"]

screen_w, screen_h = pyautogui.size()
NOSE_IDX = 1

# ===================== STATES =====================
mouse_mode = True
keyboard_mode = False
cursor_active = True
drag_active = False

prev_cursor = None
last_switch_time = 0
SWITCH_COOLDOWN = 1.5

COMMAND_PREFIX = "so "

# ===================== KEY MAP =====================
KEY_MAP = {
    "enter":"enter","back":"backspace","tab":"tab","space":"space",
    "escape":"esc","delete":"delete",
    "up":"up","down":"down","left":"left","right":"right",
    "control":"ctrl","ctrl":"ctrl","shift":"shift","alt":"alt",
    "zero":"0","one":"1","two":"2","three":"3","four":"4",
    "five":"5","six":"6","seven":"7","eight":"8","nine":"9",
    "at":"@","hash":"#","dollar":"$","percent":"%",
    "underscore":"_","minus":"-","plus":"+","equal":"=",
    "slash":"/","backslash":"\\","dot":".","comma":",",
    "question":"?","exclamation":"!","colon":":","semicolon":";"
}

# ===================== UTIL =====================
def clamp(val, minv, maxv):
    return max(minv, min(val, maxv))

def smooth(prev, cur, alpha):
    if prev is None:
        return cur
    return (
        prev[0] + alpha * (cur[0] - prev[0]),
        prev[1] + alpha * (cur[1] - prev[1])
    )

# ===================== SENSITIVITY HELPER =====================
def apply_sensitivity(new_factor):
    """
    Switch SCALING_FACTOR without letting the cursor jump or snap to centre.
    Reads the real OS cursor position and re-anchors prev_cursor there.
    """
    global SCALING_FACTOR, prev_cursor
    SCALING_FACTOR = new_factor
    if prev_cursor is not None:
        actual_x, actual_y = pyautogui.position()
        prev_cursor = (
            float(clamp(actual_x, 5, screen_w - 5)),
            float(clamp(actual_y, 5, screen_h - 5))
        )

# ===================== VOICE SETUP =====================
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_q.put(bytes(indata))

# ===================== KEYBOARD EXECUTION =====================
def execute_keyboard(text):
    """Execute a keyboard command. Returns True if a command was matched, False otherwise."""
    text = text.replace(COMMAND_PREFIX, "").strip()

    shortcuts = {
        "select all":"ctrl+a",
        "copy":"ctrl+c",
        "paste":"ctrl+v",
        "cut":"ctrl+x",
        "undo":"ctrl+z",
        "redo":"ctrl+y",
        "save":"ctrl+s"
    }

    if text in shortcuts:
        keyboard.press_and_release(shortcuts[text])
        return True

    text = text.replace("plus", "+").replace("and", "+")
    parts = [p.strip() for p in text.split("+")]

    keys = []
    chars = ""

    for part in parts:
        if part in KEY_MAP:
            val = KEY_MAP[part]
            if val in ["ctrl","shift","alt","enter","tab",
                       "esc","backspace","delete","up","down","left","right"]:
                keys.append(val)
            else:
                chars += val
        elif len(part) == 1:
            keys.append(part)

    if keys:
        keyboard.press_and_release("+".join(keys))
        return True

    if chars:
        pyautogui.write(chars, interval=0.02)
        return True

    return False

# ===================== MAIN =====================
def main():
    global mouse_mode, keyboard_mode
    global cursor_active, SCALING_FACTOR
    global prev_cursor, last_switch_time, drag_active

    print("🎤 AI HCI SYSTEM READY")
    print("👉 Tilt RIGHT = Mouse Mode")
    print("👈 Tilt LEFT  = Keyboard Mode")
    print("🎙 Say 'exit system' to close")
    print("🔻 Default Sensitivity: LOW (say 'low', 'medium', 'high' to change)")
    print("🖱 Mouse Mode: say 'click', 'right click', 'double click', 'drag', 'stop drag'")
    print("⌨  Keyboard Mode: speak normally to type | say 'so <command>' for actions")
    print("   └─ 'so back' = 3 backspaces\n")

    cap = cv2.VideoCapture(0)
    mp_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=4000,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            res = mp_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # ================= HEAD TILT MODE SWITCH =================
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark

                left_eye  = lm[33]
                right_eye = lm[263]

                x1, y1 = int(left_eye.x  * w), int(left_eye.y  * h)
                x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                current_time = time.time()

                if current_time - last_switch_time > SWITCH_COOLDOWN:

                    if angle > 15 and not mouse_mode:
                        mouse_mode    = True
                        keyboard_mode = False
                        cursor_active = True
                        prev_cursor   = None
                        last_switch_time = current_time
                        print("🖱 Mouse Mode Activated")

                    elif angle < -15 and not keyboard_mode:
                        keyboard_mode = True
                        mouse_mode    = False
                        last_switch_time = current_time
                        print("⌨ Keyboard Mode Activated")

                # ================= CURSOR MOVEMENT =================
                if mouse_mode and cursor_active:

                    nx, ny = lm[NOSE_IDX].x, lm[NOSE_IDX].y

                    raw_x = nx * screen_w * SCALING_FACTOR - (SCALING_FACTOR - 1) * screen_w / 2
                    raw_y = ny * screen_h * SCALING_FACTOR - (SCALING_FACTOR - 1) * screen_h / 2

                    raw = (raw_x, raw_y)

                    prev_cursor = smooth(prev_cursor, raw, SMOOTHING_ALPHA)

                    pyautogui.moveTo(
                        clamp(int(prev_cursor[0]), 5, screen_w - 5),
                        clamp(int(prev_cursor[1]), 5, screen_h - 5),
                        duration=0
                    )

            # ================= VOICE COMMANDS =================
            if not audio_q.empty():
                data = audio_q.get()

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text   = result.get("text", "").lower()

                    if not text:
                        continue

                    print("🎙", text)

                    # ===== EXIT SYSTEM =====
                    if text == "exit system":
                        print("🛑 Shutting Down System...")
                        break

                    # ===== CURSOR CONTROL =====
                    elif text == "start cursor":
                        cursor_active = True
                        prev_cursor   = None
                        print("▶ Cursor Started")

                    elif text == "stop cursor":
                        cursor_active = False
                        print("⏹ Cursor Stopped")

                    # ===== SENSITIVITY COMMANDS =====
                    elif text in ("high", "hey", "i", "fast"):
                        apply_sensitivity(SENSITIVITY["high"])
                        print("🔺 Sensitivity: HIGH")

                    elif text == "medium":
                        apply_sensitivity(SENSITIVITY["medium"])
                        print("🔹 Sensitivity: MEDIUM")

                    elif text in ("low", "slow", "hello"):
                        apply_sensitivity(SENSITIVITY["low"])
                        print("🔻 Sensitivity: LOW")

                    # ===== MOUSE COMMANDS (no "so" prefix needed) =====
                    elif mouse_mode and text == "click":
                        pyautogui.click()
                        print("🖱 Click")

                    elif mouse_mode and text == "right click":
                        pyautogui.rightClick()
                        print("🖱 Right Click")

                    elif mouse_mode and text == "double click":
                        pyautogui.doubleClick()
                        print("🖱 Double Click")

                    # FIX: wrap the OR alternatives in parentheses so the
                    # whole condition is evaluated correctly.
                    # Old (broken): mouse_mode and text == "drag" or "brad" and not drag_active
                    #   → "brad" is a non-empty string, always True → drag fires on every command
                    # Fixed: mouse_mode and (text == "drag" or text == "brad") and not drag_active
                    elif mouse_mode and (text == "drag" or text == "brad") and not drag_active:
                        pyautogui.mouseDown()
                        drag_active = True
                        print("🖱 Drag Started")

                    elif mouse_mode and (text == "stop drag" or text == "stop brad") and drag_active:
                        pyautogui.mouseUp()
                        drag_active = False
                        print("🖱 Drag Stopped")

                    # ===== KEYBOARD COMMANDS ("so" prefix required) =====
                    elif keyboard_mode and text.startswith(COMMAND_PREFIX):
                        cmd = text.replace(COMMAND_PREFIX, "").strip()

                        # "so back" → press backspace 3 times
                        if cmd == "back":
                            for _ in range(3):
                                keyboard.press_and_release("backspace")
                            print("⌫⌫⌫ Triple Backspace")

                        else:
                            executed = execute_keyboard(text)
                            if not executed:
                                print(f"⚠ Unknown keyboard command: 'so {cmd}' — ignored (not typed)")

                    elif keyboard_mode:
                        # Plain speech (no "so " prefix) → type it normally
                        pyautogui.write(text + " ", interval=0.02)

            # ================= DISPLAY MODE ON FRAME =================
            mode_text  = "MOUSE MODE" if mouse_mode else "KEYBOARD MODE"
            sens_label = [k for k, v in SENSITIVITY.items() if v == SCALING_FACTOR]
            sens_text  = sens_label[0].upper() if sens_label else "CUSTOM"

            cv2.putText(frame, mode_text, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if mouse_mode else (255, 0, 0), 2)

            cv2.putText(frame, f"Sensitivity: {sens_text}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 255), 2)

            cv2.imshow("AI HCI System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ System Closed Safely")

if __name__ == "__main__":
    main()