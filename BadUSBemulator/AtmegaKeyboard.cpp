#include <Keyboard.h>

void setup() {
  Serial1.begin(115200); // UART for connection Orange Pi
  Keyboard.begin();
}

void loop() {
  if (Serial1.available() > 0) {
    String data = Serial1.readStringUntil('\n');
    char action = data.charAt(0);
    int keycode = data.substring(2).toInt();

    if (action == 'P') {
      Keyboard.press(keycode);
    } else if (action == 'R') {
      Keyboard.release(keycode);
    }
  }
}
