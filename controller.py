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
    "medium": 5.5,
    "high":   9.0
}

SCALING_FACTOR = SENSITIVITY["high"]

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

last_click_time = 0
CLICK_COOLDOWN = 1.5

COMMAND_PREFIX = "so"

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

# ===================== COMMAND SYNONYMS =====================
COMMAND_SYNONYMS = {
    # --- exit ---
    "exit system": [
        "exit system", "exit sistem", "exit this system", "exit the system",
        "egg system", "exist system", "exit systems", "exit sistem"
    ],
    # --- cursor control ---
    "start cursor": [
        "start cursor", "start cursor please", "start the cursor",
        "store cursor", "start coarser", "start curser"
    ],
    "stop cursor": [
        "stop cursor", "stop the cursor", "stop curser",
        "shop cursor", "top cursor", "stop coarser"
    ],
    # --- sensitivity ---
    "high": [
        "high", "hey", "i", "fast", "hi", "hai", "hye",
        "hide", "hire", "higher", "hike", "height", "tied"
    ],
    "medium": [
        "medium", "median", "medium speed", "media", "meet him",
        "met him", "medium sensitivity", "need him"
    ],
    "low": [
        "low", "slow", "hello", "below", "flow", "glow",
        "lo", "law", "allow", "lower", "low speed"
    ],
    "click": [
        "click", "clic", "klick", "click please", "creek",
        "lick", "tick", "brick", "flick", "sick",
        "clique", "klik", "clk", "clek", "clik",
        "a click", "one click", "do click", "please click",
        "quick", "quake", "clock", "clip", "clicked",
        "clique", "clack", "cluck", "cloc", "clee",
        "glik", "glic", "glick", "klic", "klek",
        "click it", "click now", "just click",
        "the it", "digg", "dig", "link", "lik", "lick it",
        "dig it", "thick", "tic", "dic", "dik", "tik",
        "chick", "chic", "chik", "kik", "kic", "keck",
        "it", "get", "kit", "bit", "hit", "sit", "fit", "mit",
        "pick", "kick", "nick", "wick", "rick", "dick", "tic tac"
    ],
    "right click": [
        "right click", "right clic", "write click", "right klick",
        "right creek", "rice click", "ride click", "right lick",
        "right quick", "right klik", "right clique", "right clek",
        "right clip", "right clack", "write clic", "write klick",
        "right click please", "do right click", "right clicked"
    ],
    "double click": [
        "double click", "double clic", "double klick", "durable click",
        "trouble click", "double creek", "double lick", "dub click",
        "double klik", "double clique", "double clek", "double clip",
        "double clack", "double quick", "to double click", "double clicked",
        "double click please", "dbl click", "dubble click", "doble click"
    ],
    "drag": [
        "drag", "brad", "drug", "drake", "track", "dreg",
        "drag please", "drab", "rag", "craig", "brag"
    ],
    "stop drag": [
        "stop", "top", "shop", "cop", "mop", "drop",
        "stop it", "stop drag", "stopped", "stomp"
    ],
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

# ===================== PHONETIC HELPER =====================
def _soundex(word):
    """Basic Soundex phonetic encoding."""
    word = word.upper().strip()
    if not word:
        return ""
    codes = {'BFPV': '1', 'CGJKQSXYZ': '2', 'DT': '3',
             'L': '4', 'MN': '5', 'R': '6'}
    result = word[0]
    prev = '0'
    for ch in word[1:]:
        code = '0'
        for letters, digit in codes.items():
            if ch in letters:
                code = digit
                break
        if code != '0' and code != prev:
            result += code
        prev = code
    return (result + "000")[:4]

# ===================== COMMAND MATCHER =====================
def match_command(text, command_name):
    text = text.strip().lower()
    synonyms = [s.strip().lower() for s in COMMAND_SYNONYMS.get(command_name, [])]
    if text in synonyms:
        return True
    if " " not in command_name and " " not in text:
        target_codes = set(_soundex(s) for s in synonyms if " " not in s)
        if _soundex(text) in target_codes:
            return True
    return False

# ===================== SENSITIVITY HELPER =====================
def apply_sensitivity(new_factor):
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

# ===================== CLICK HANDLER WITH COOLDOWN =====================
def handle_click_commands(text):
    global last_click_time, drag_active

    now = time.time()

    if mouse_mode and match_command(text, "double click"):
        if now - last_click_time > CLICK_COOLDOWN:
            pyautogui.doubleClick()
            last_click_time = now
            print("🖱 Double Click")
        return True

    elif mouse_mode and match_command(text, "right click"):
        if now - last_click_time > CLICK_COOLDOWN:
            pyautogui.rightClick()
            last_click_time = now
            print("🖱 Right Click")
        return True

    elif mouse_mode and match_command(text, "click"):
        if now - last_click_time > CLICK_COOLDOWN:
            pyautogui.click()
            last_click_time = now
            print("🖱 Click")
        return True

    elif mouse_mode and match_command(text, "drag") and not drag_active:
        if now - last_click_time > CLICK_COOLDOWN:
            pyautogui.mouseDown()
            drag_active = True
            last_click_time = now
            print("🖱 Drag Started")
        return True

    elif mouse_mode and match_command(text, "stop drag") and drag_active:
        pyautogui.mouseUp()
        drag_active = False
        print("🖱 Drag Stopped")
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
    print("🔻 Default Sensitivity: HIGH (say 'low', 'medium', 'high' to change)")
    print("🖱 Mouse Mode: say 'click', 'right click', 'double click', 'drag', 'stop drag'")
    print("⌨  Keyboard Mode: speak normally to type | say 'so <command>' for actions")
    print("   └─ 'so back' = 3 backspaces\n")

    cap = cv2.VideoCapture(0)
    mp_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    # FIX 1 — blocksize reduced: 4000→1600 (100ms chunks) for faster command response
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=1600,
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

                    # FIX 2 — UNSTICKY EDGES:
                    # Clamp the RAW position before smoothing so prev_cursor never
                    # accumulates an out-of-bounds value. Without this, prev_cursor
                    # drifts deep into the clamped wall and the cursor won't move
                    # back until enough smooth() iterations "unwind" that drift.
                    raw_x = clamp(raw_x, 5, screen_w - 5)
                    raw_y = clamp(raw_y, 5, screen_h - 5)
                    raw = (raw_x, raw_y)

                    prev_cursor = smooth(prev_cursor, raw, SMOOTHING_ALPHA)

                    pyautogui.moveTo(
                        clamp(int(prev_cursor[0]), 5, screen_w - 5),
                        clamp(int(prev_cursor[1]), 5, screen_h - 5),
                        duration=0
                    )

            # ================= VOICE COMMANDS =================
            # FIX 3 — DRAIN the queue each frame so audio never falls behind.
            # Previously a single audio_q.empty() check meant chunks pile up
            # during heavy CPU frames, causing recognition to lag by seconds.
            while not audio_q.empty():
                data = audio_q.get()

                # Check partial results for low-latency click detection
                partial_raw = json.loads(recognizer.PartialResult())
                partial_text = partial_raw.get("partial", "").strip().lower()

                if partial_text and mouse_mode:
                    handle_click_commands(partial_text)

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text   = result.get("text", "").lower()

                    if not text:
                        continue

                    print("🎙", text)

                    # ===== EXIT SYSTEM =====
                    if match_command(text, "exit system"):
                        print("🛑 Shutting Down System...")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                    # ===== CURSOR CONTROL =====
                    elif match_command(text, "start cursor"):
                        cursor_active = True
                        prev_cursor   = None
                        print("▶ Cursor Started")

                    elif match_command(text, "stop cursor"):
                        cursor_active = False
                        print("⏹ Cursor Stopped")

                    # ===== SENSITIVITY COMMANDS =====
                    elif match_command(text, "high"):
                        apply_sensitivity(SENSITIVITY["high"])
                        print("🔺 Sensitivity: HIGH")

                    elif match_command(text, "medium"):
                        apply_sensitivity(SENSITIVITY["medium"])
                        print("🔹 Sensitivity: MEDIUM")

                    elif match_command(text, "low"):
                        apply_sensitivity(SENSITIVITY["low"])
                        print("🔻 Sensitivity: LOW")

                    # ===== MOUSE COMMANDS — routed through cooldown-guarded handler =====
                    elif not handle_click_commands(text):

                        # ===== KEYBOARD COMMANDS ("so" prefix required) =====
                        if keyboard_mode and text.startswith(COMMAND_PREFIX):
                            cmd = text.replace(COMMAND_PREFIX, "").strip()

                            if cmd == "back":
                                for _ in range(4):
                                    keyboard.press_and_release("backspace")
                                print("⌫⌫⌫ Triple Backspace")

                            else:
                                executed = execute_keyboard(text)
                                if not executed:
                                    print(f"⚠ Unknown keyboard command: 'so {cmd}' — ignored (not typed)")

                        elif keyboard_mode:
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