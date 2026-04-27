import sys, mss, cv2, numpy as np
sys.path.insert(0, ".")
from color_detect import quadrant_colors, is_kahoot_question

with mss.mss() as sct:
    print("Tilgængelige skærme:")
    for i, m in enumerate(sct.monitors):
        print(f"  monitor[{i}] = {m}")

    for i in range(1, len(sct.monitors)):
        shot = sct.grab(sct.monitors[i])
        frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
        cv2.imwrite(f"debug_screen_{i}.png", frame)
        print(f"\n--- Monitor {i} ({frame.shape[1]}x{frame.shape[0]}) ---")
        print("kahoot_quadrant_coverage:", quadrant_colors(frame))
        print("is_kahoot_question:", is_kahoot_question(frame))
        print(f"Gemte: debug_screen_{i}.png")