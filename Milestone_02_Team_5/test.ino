#define LEFT_MOTOR11 8
#define LEFT_MOTOR12 11
#define RIGHT_MOTOR11 2
#define RIGHT_MOTOR12 1
#define LEFT_MOTOR21 12
#define LEFT_MOTOR22 13
#define RIGHT_MOTOR21 6
#define RIGHT_MOTOR22 7

// PWM pins
#define pwm1 3  // Left Motor 1
#define pwm2 5  // Right Motor 1
#define pwm3 9  // Left Motor 2
#define pwm4 10 // Right Motor 2

int pwmValue = 0;  // Default PWM

void setup() {
  pinMode(LEFT_MOTOR11, OUTPUT);
  pinMode(LEFT_MOTOR12, OUTPUT);
  pinMode(RIGHT_MOTOR11, OUTPUT);
  pinMode(RIGHT_MOTOR12, OUTPUT);
  pinMode(LEFT_MOTOR21, OUTPUT);
  pinMode(LEFT_MOTOR22, OUTPUT);
  pinMode(RIGHT_MOTOR21, OUTPUT);
  pinMode(RIGHT_MOTOR22, OUTPUT);

  pinMode(pwm1, OUTPUT);
  pinMode(pwm2, OUTPUT);
  pinMode(pwm3, OUTPUT);
  pinMode(pwm4, OUTPUT);

  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');  // Expecting something like R120
    if (input.length() >= 2) {
      char command = input.charAt(0);             // Direction (R/L/F/B/S)
      int value = input.substring(1).toInt();     // PWM value

      pwmValue = constrain(value, 0, 255);         // Keep within bounds

      switch (command) {
        case 'L': turnLeft(pwmValue); break;
        case 'R': turnRight(pwmValue); break;
        case 'F': forward(pwmValue); break;
        case 'B': reverse(pwmValue); break;
        case 'S': stopCar(); break;
      }
    }
  }
}

// === Movement Functions with PWM ===

void forward(int pwm) {
  digitalWrite(LEFT_MOTOR11, HIGH);
  digitalWrite(LEFT_MOTOR12, LOW);
  digitalWrite(RIGHT_MOTOR11, HIGH);
  digitalWrite(RIGHT_MOTOR12, LOW);
  digitalWrite(LEFT_MOTOR21, HIGH);
  digitalWrite(LEFT_MOTOR22, LOW);
  digitalWrite(RIGHT_MOTOR21, HIGH);
  digitalWrite(RIGHT_MOTOR22, LOW);

  analogWrite(pwm1, pwm);
  analogWrite(pwm2, pwm);
  analogWrite(pwm3, pwm);
  analogWrite(pwm4, pwm);

  Serial.println("Action: FORWARD");
}

void reverse(int pwm) {
  digitalWrite(LEFT_MOTOR11, LOW);
  digitalWrite(LEFT_MOTOR12, HIGH);
  digitalWrite(RIGHT_MOTOR11, LOW);
  digitalWrite(RIGHT_MOTOR12, HIGH);
  digitalWrite(LEFT_MOTOR21, LOW);
  digitalWrite(LEFT_MOTOR22, HIGH);
  digitalWrite(RIGHT_MOTOR21, LOW);
  digitalWrite(RIGHT_MOTOR22, HIGH);

  analogWrite(pwm1, pwm);
  analogWrite(pwm2, pwm);
  analogWrite(pwm3, pwm);
  analogWrite(pwm4, pwm);

  Serial.println("Action: REVERSE");
}

void turnLeft(int pwm) {
  digitalWrite(LEFT_MOTOR11, LOW);
  digitalWrite(LEFT_MOTOR12, HIGH);
  digitalWrite(RIGHT_MOTOR11, HIGH);
  digitalWrite(RIGHT_MOTOR12, LOW);
  digitalWrite(LEFT_MOTOR21, LOW);
  digitalWrite(LEFT_MOTOR22, HIGH);
  digitalWrite(RIGHT_MOTOR21, HIGH);
  digitalWrite(RIGHT_MOTOR22, LOW);

  analogWrite(pwm1, pwm);
  analogWrite(pwm2, pwm);
  analogWrite(pwm3, pwm);
  analogWrite(pwm4, pwm);

  Serial.println("Action: TURN LEFT");
}

void turnRight(int pwm) {
  digitalWrite(LEFT_MOTOR11, HIGH);
  digitalWrite(LEFT_MOTOR12, LOW);
  digitalWrite(RIGHT_MOTOR11, LOW);
  digitalWrite(RIGHT_MOTOR12, HIGH);
  digitalWrite(LEFT_MOTOR21, HIGH);
  digitalWrite(LEFT_MOTOR22, LOW);
  digitalWrite(RIGHT_MOTOR21, LOW);
  digitalWrite(RIGHT_MOTOR22, HIGH);

  analogWrite(pwm1, pwm);
  analogWrite(pwm2, pwm);
  analogWrite(pwm3, pwm);
  analogWrite(pwm4, pwm);

  Serial.println("Action: TURN RIGHT");
}

void stopCar() {
  digitalWrite(LEFT_MOTOR11, LOW);
  digitalWrite(LEFT_MOTOR12, LOW);
  digitalWrite(RIGHT_MOTOR11, LOW);
  digitalWrite(RIGHT_MOTOR12, LOW);
  digitalWrite(LEFT_MOTOR21, LOW);
  digitalWrite(LEFT_MOTOR22, LOW);
  digitalWrite(RIGHT_MOTOR21, LOW);
  digitalWrite(RIGHT_MOTOR22, LOW);

  analogWrite(pwm1, 0);
  analogWrite(pwm2, 0);
  analogWrite(pwm3, 0);
  analogWrite(pwm4, 0);

  Serial.println("Action: STOP");
}
