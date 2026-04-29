import sys
import os
import time
import serial
import re
import numpy as np
import random

# UART Settings (match Serial1.begin(115200) in Arduino)
UART_PORT = "/dev/ttyUSB0"
UART_BAUD = 115200

# Mapping special keys to Arduino Keyboard.h codes
# In Arduino: KEY_LEFT_CTRL = 0x80 (128), KEY_ESC = 0xB1 (177), etc.
KEYMAP = {
    '[F1]': 194, '[F2]': 195, '[F3]': 196, '[F4]': 197, '[F5]': 198, '[F6]': 199,
    '[F7]': 200, '[F8]': 201, '[F9]': 202, '[F10]': 203, '[F11]': 204, '[F12]': 205,
    '[ESC]': 177, '[ALT]': 130, '[PTRSCR]': 206, '[TAB]': 179, '[SPACE]': 32,
    '[WINDOW]': 131, '[CTRL]': 128, '[DELETE]': 212, '[ENTER]': 176, '[SHIFT]': 129
}

def parse_file(filepath):
    """Parses the input script file for injection tokens"""
    if not os.path.exists(filepath):
        print(f"\033[91mError: File {filepath} not found!\033[0m")
        sys.exit(1)
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    tokens = []
    # Added the plus sign (\+) to the special key regex to support combinations
    pattern = re.compile(r'(" delay\(\d+\)")|(delay\((\d+)\))|(\[[A-Za-z0-9_+\-]+\])|(.)', re.DOTALL)

    for match in pattern.finditer(content):
        if match.group(1):  # Literal string like " delay(100)"
            literal = match.group(1)[1:-1]
            for char in literal:
                tokens.append(('CHAR', ord(char)))
                
        elif match.group(2):  # Scripted delay command: delay(X)
            tokens.append(('DELAY', int(match.group(3))))
            
        elif match.group(4):  # Special keys or combinations: [CTRL+ALT+t] or [F1]
            content_inside = match.group(4)[1:-1] # Strip brackets
            keys = content_inside.split('+')
            combo_codes = []
            
            for k in keys:
                k_upper = f"[{k.upper()}]"
                if k_upper in KEYMAP:
                    combo_codes.append(KEYMAP[k_upper])
                elif len(k) == 1:
                    # For regular letters in a combo (e.g., 't' in [CTRL+t])
                    # We use lower case so Arduino doesn't auto-press Shift
                    combo_codes.append(ord(k.lower())) 
                else:
                    print(f"\033[91mParsing error: Unknown key '{k}' in combo '{match.group(4)}'\033[0m")
                    sys.exit(1)
            
            # If it's a single key, add as SPECIAL; if multiple, add as COMBO
            if len(combo_codes) == 1:
                tokens.append(('SPECIAL', combo_codes[0]))
            else:
                tokens.append(('COMBO', combo_codes))
                
        elif match.group(5):  # Standard single character
            char = match.group(5)
            if 32 <= ord(char) <= 126 or char in ('\n', '\r'):
                tokens.append(('CHAR', ord(char)))
            else:
                print(f"\033[91mParsing error: Unrecognized character '{char}'\033[0m")
                sys.exit(1)
                
    return tokens

# ==============================================================================
# ATTACK GENERATORS
# ==============================================================================

def get_machine_gun(num_events, is_custom):
    if is_custom:
        base_val = float(input("Enter base speed (ms) [e.g., 8]: "))
    else:
        base_val = random.randint(4, 12)
    dwells = np.clip(np.random.normal(base_val, 0.5, num_events), 1, None)
    flights = np.clip(np.random.normal(base_val, 0.5, num_events), 1, None)
    return dwells, flights

def get_the_robot(num_events, is_custom):
    if is_custom:
        val = float(input("Enter exact timing (ms) [e.g., 40]: "))
    else:
        val = random.randint(1, 80)
    return np.full(num_events, val), np.full(num_events, val)

def get_gaussian_faker(num_events, is_custom):
    if is_custom:
        mean_d = float(input("Enter Dwell Mean (ms): "))
        std_d = float(input("Enter Dwell StdDev (Sigma): "))
        mean_f = float(input("Enter Flight Mean (ms): "))
        std_f = float(input("Enter Flight StdDev (Sigma): "))
    else:
        mean_d, std_d = random.randint(70, 130), random.randint(10, 50)
        mean_f, std_f = random.randint(80, 150), random.randint(15, 60)

    dwells = np.clip(np.random.normal(mean_d, std_d, num_events), 10, None)
    flights = np.clip(np.random.normal(mean_f, std_f, num_events), 10, None)
    return dwells, flights

def get_uniform_jitter(num_events, is_custom):
    if is_custom:
        low_d = float(input("Enter Min Dwell: "))
        high_d = float(input("Enter Max Dwell: "))
        low_f = float(input("Enter Min Flight: "))
        high_f = float(input("Enter Max Flight: "))
    else:
        low_d, high_d = random.randint(5, 55), random.randint(60, 150)
        low_f, high_f = random.randint(5, 55), random.randint(60, 200)

    dwells = np.random.uniform(low_d, high_d, num_events)
    flights = np.random.uniform(low_f, high_f, num_events)
    return dwells, flights

def get_burst_mode(num_events, is_custom):
    if is_custom:
        b_min = int(input("Enter Min Burst Size: "))
        b_max = int(input("Enter Max Burst Size: "))
        p_min = float(input("Enter Min Pause (ms): "))
        p_max = float(input("Enter Max Pause (ms): "))
    else:
        b_min, b_max = 10, 30
        p_min, p_max = 200, 1000

    dwells, flights = [], []
    while len(dwells) < num_events:
        burst_size = random.randint(b_min, b_max)
        pause_time = random.randint(int(p_min), int(p_max))

        for _ in range(burst_size):
            dwells.append(random.randint(5, 15))
            flights.append(random.randint(5, 15))

        if len(flights) > 0:
            flights[-1] = pause_time

    return np.array(dwells[:num_events]), np.array(flights[:num_events])

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 host_platform.py <input_file.txt>")
        sys.exit(1)

    filepath = sys.argv[1]
    tokens = parse_file(filepath)

    # Count only actual key events (ignore algorithmic delay commands)
    num_events = sum(1 for t in tokens if t[0] in ('CHAR', 'SPECIAL', 'COMBO'))
    print(f"File parsed successfully. Total events to inject: {num_events}")

    print("\n--- Select Attack Profile ---")
    print("1. Machine Gun (Very fast, low variance)")
    print("2. The Robot (Perfectly uniform)")
    print("3. Gaussian Faker (Human-like normal distribution)")
    print("4. Uniform Jitter (Random chaos within bounds)")
    print("5. Burst Mode (Fast batches with pauses)")

    choice = input("Select attack [1-5]: ")
    mode = input("Use Custom parameters? (y/n): ").strip().lower() == 'y'

    attacks = {
        '1': get_machine_gun,
        '2': get_the_robot,
        '3': get_gaussian_faker,
        '4': get_uniform_jitter,
        '5': get_burst_mode
    }

    if choice not in attacks:
        print("\033[91mInvalid attack selection!\033[0m")
        sys.exit(1)

    # Generate timing arrays
    dwells, flights = attacks[choice](num_events, mode)

    print(f"\n🚀 Starting injection over {UART_PORT}...")
    try:
        ser = serial.Serial(UART_PORT, UART_BAUD, timeout=1)
        time.sleep(2) # Wait for serial port initialization
    except Exception as e:
        print(f"\033[91mFailed to open {UART_PORT}: {e}\033[0m")
        sys.exit(1)

    event_idx = 0

    for token_type, val in tokens:
        if token_type == 'DELAY':
            # Algorithm-defined hard delay
            time.sleep(val / 1000.0)
            
        elif token_type in ('CHAR', 'SPECIAL'):
            flight = flights[event_idx]
            dwell = dwells[event_idx]
            event_idx += 1
            
            time.sleep(flight / 1000.0)
            ser.write(f"P,{val}\n".encode('utf-8'))
            time.sleep(dwell / 1000.0)
            ser.write(f"R,{val}\n".encode('utf-8'))
            
        elif token_type == 'COMBO':
            # 'val' is a list of key codes
            flight = flights[event_idx]
            dwell = dwells[event_idx]
            event_idx += 1
            
            time.sleep(flight / 1000.0)
            
            # Press all keys in the combination
            for code in val:
                ser.write(f"P,{code}\n".encode('utf-8'))
                time.sleep(0.01) # Give UART 10ms to prevent buffer overflow
                
            # Keep them pressed
            time.sleep(dwell / 1000.0)
            
            # Release all keys in reverse order (mimics human behavior)
            for code in reversed(val):
                ser.write(f"R,{code}\n".encode('utf-8'))
                time.sleep(0.01)

    print("✅ Attack completed successfully.")
    ser.close()

if __name__ == "__main__":
    main()
