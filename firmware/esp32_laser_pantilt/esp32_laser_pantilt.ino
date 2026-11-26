#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>
#include <ESPmDNS.h>

// ==================================================
// WIFI SETTINGS (STATION MODE - MOBILE HOTSPOT)
// ==================================================

const char* STA_SSID = "mi11TPro";
const char* STA_PASSWORD = "12345678";

WebServer server(80);

// ==================================================
// HARDWARE PINS
// ==================================================

const int PAN_PIN = 18;
const int TILT_PIN = 19;
const int LASER_PIN = 23;

// ==================================================
// SERVO SETTINGS
// ==================================================

const int PAN_HOME = 90;
const int TILT_HOME = 90;

const int MIN_ANGLE = 10;
const int MAX_ANGLE = 180;

// Maximum commanded angle change per update.
const int SERVO_STEP = 2;

// Update interval in milliseconds.
const unsigned long SERVO_UPDATE_MS = 20;

Servo panServo;
Servo tiltServo;

int panCurrent = PAN_HOME;
int tiltCurrent = TILT_HOME;

int panTarget = PAN_HOME;
int tiltTarget = TILT_HOME;

bool laserOn = false;
bool emergencyStopped = false;

unsigned long lastServoUpdate = 0;

// ==================================================
// DASHBOARD HTML
// ==================================================

const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PoolBot Controller</title>
<style>
* { box-sizing: border-box; }
body {
  margin: 0; padding: 20px;
  font-family: Arial, sans-serif;
  background: #101820; color: #f4f7fa;
}
.container { max-width: 520px; margin: auto; }
h1 { text-align: center; margin-bottom: 5px; }
.subtitle { text-align: center; color: #aab7c4; margin-bottom: 25px; }
.card {
  background: #1c2833; border: 1px solid #354656;
  border-radius: 14px; padding: 18px; margin-bottom: 16px;
}
.row { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.angle { font-size: 22px; font-weight: bold; }
input[type=range] { width: 100%; margin: 22px 0 10px; }
button {
  width: 100%; padding: 14px; margin-top: 10px;
  border: none; border-radius: 9px; font-size: 16px; font-weight: bold; cursor: pointer;
}
.center { background: #2471a3; color: white; }
.laserOn { background: #287d42; color: white; }
.laserOff { background: #566573; color: white; }
.stop { background: #c0392b; color: white; font-size: 18px; }
.status { text-align: center; color: #58d68d; font-weight: bold; }
.note { color: #b9c6d2; font-size: 13px; line-height: 1.5; }
</style>
</head>
<body>
<div class="container">
<h1>PoolBot</h1>
<div class="subtitle">ESP32 Pan-Tilt Controller (Dual Wi-Fi)</div>

<div class="card">
  <div class="row">
    <strong>Connection</strong>
    <span id="connection">Connecting...</span>
  </div>
  <p class="status" id="systemStatus">Initializing</p>
</div>

<div class="card">
  <div class="row">
    <strong>Pan: Left / Right</strong>
    <span class="angle" id="panValue">90°</span>
  </div>
  <input id="panSlider" type="range" min="10" max="180" value="90" oninput="previewAngle('pan')" onchange="setAngle('pan')"/>
</div>

<div class="card">
  <div class="row">
    <strong>Tilt: Up / Down</strong>
    <span class="angle" id="tiltValue">90°</span>
  </div>
  <input id="tiltSlider" type="range" min="20" max="160" value="90" oninput="previewAngle('tilt')" onchange="setAngle('tilt')"/>
</div>

<div class="card">
  <button class="center" onclick="centerServos()">CENTER BOTH SERVOS</button>
  <button class="laserOn" onclick="setLaser(true)">LASER ON</button>
  <button class="laserOff" onclick="setLaser(false)">LASER OFF</button>
  <button class="stop" onclick="stopSystem()">EMERGENCY STOP</button>
</div>

<div class="card">
  <strong>Current system state</strong>
  <p>Pan: <span id="panState">90</span>°</p>
  <p>Tilt: <span id="tiltState">90</span>°</p>
  <p>Laser: <span id="laserState">OFF</span></p>
  <p class="note">
    STOP disables commanded servo movement and turns off the laser output. It does not disconnect power.
  </p>
</div>

</div>

<script>
let busy = false;
function previewAngle(axis) {
  const slider = document.getElementById(axis === 'pan' ? 'panSlider' : 'tiltSlider');
  document.getElementById(axis === 'pan' ? 'panValue' : 'tiltValue').textContent = slider.value + '°';
}
async function request(path) {
  try {
    const response = await fetch(path, { cache: 'no-store' });
    if (!response.ok) throw new Error('HTTP ' + response.status);
    return await response.text();
  } catch (error) {
    document.getElementById('connection').textContent = 'Disconnected';
    document.getElementById('connection').style.color = '#ff7675';
    throw error;
  }
}
async function setAngle(axis) {
  if (busy) return;
  const slider = document.getElementById(axis === 'pan' ? 'panSlider' : 'tiltSlider');
  try {
    await request('/move?axis=' + axis + '&angle=' + slider.value);
    await updateStatus();
  } catch (error) { console.error(error); }
}
async function centerServos() {
  try {
    await request('/center');
    document.getElementById('panSlider').value = 90;
    document.getElementById('tiltSlider').value = 90;
    previewAngle('pan'); previewAngle('tilt');
    await updateStatus();
  } catch (error) { console.error(error); }
}
async function setLaser(state) {
  try {
    await request('/laser?state=' + (state ? '1' : '0'));
    await updateStatus();
  } catch (error) { console.error(error); }
}
async function stopSystem() {
  try {
    busy = true;
    await request('/stop');
    await updateStatus();
  } catch (error) { console.error(error); } finally { busy = false; }
}
async function updateStatus() {
  try {
    const response = await fetch('/status', { cache: 'no-store' });
    if (!response.ok) throw new Error('Status unavailable');
    const data = await response.json();
    document.getElementById('connection').textContent = 'Connected';
    document.getElementById('connection').style.color = '#58d68d';
    document.getElementById('systemStatus').textContent = data.stopped ? 'STOPPED' : 'Ready';
    document.getElementById('panState').textContent = data.pan;
    document.getElementById('tiltState').textContent = data.tilt;
    document.getElementById('laserState').textContent = data.laser ? 'ON' : 'OFF';
  } catch (error) {
    document.getElementById('connection').textContent = 'Disconnected';
  }
}
setInterval(updateStatus, 2000);
updateStatus();
</script>
</body>
</html>
)rawliteral";

// ==================================================
// SERVO MOVEMENT
// ==================================================

int approachAngle(int current, int target) {
  if (current < target) return min(current + SERVO_STEP, target);
  if (current > target) return max(current - SERVO_STEP, target);
  return current;
}

void updateServos() {
  if (emergencyStopped) return;
  unsigned long now = millis();
  if (now - lastServoUpdate < SERVO_UPDATE_MS) return;
  lastServoUpdate = now;

  int newPan = approachAngle(panCurrent, panTarget);
  int newTilt = approachAngle(tiltCurrent, tiltTarget);

  if (newPan != panCurrent) {
    panCurrent = newPan;
    panServo.write(panCurrent);
  }
  if (newTilt != tiltCurrent) {
    tiltCurrent = newTilt;
    tiltServo.write(tiltCurrent);
  }
}

// ==================================================
// WEB HANDLERS
// ==================================================

void handleRoot() {
  server.send_P(200, "text/html", INDEX_HTML);
}

void handleMove() {
  if (emergencyStopped) {
    server.send(423, "text/plain", "Movement stopped. Use CENTER to resume.");
    return;
  }
  if (!server.hasArg("axis") || !server.hasArg("angle")) {
    server.send(400, "text/plain", "Missing axis or angle");
    return;
  }
  String axis = server.arg("axis");
  String angleText = server.arg("angle");

  for (unsigned int i = 0; i < angleText.length(); i++) {
    if (!isDigit(angleText[i])) {
      server.send(400, "text/plain", "Invalid angle");
      return;
    }
  }

  int angle = angleText.toInt();
  if (angle < MIN_ANGLE || angle > MAX_ANGLE) {
    server.send(400, "text/plain", "Angle outside safe range");
    return;
  }

  if (axis == "pan") panTarget = angle;
  else if (axis == "tilt") tiltTarget = angle;
  else {
    server.send(400, "text/plain", "Invalid axis");
    return;
  }
  server.send(200, "text/plain", "Target accepted");
}

void handleCenter() {
  emergencyStopped = false;
  panTarget = PAN_HOME;
  tiltTarget = TILT_HOME;
  digitalWrite(LASER_PIN, LOW);
  laserOn = false;
  server.send(200, "text/plain", "Centering servos");
}

void handleLaser() {
  if (!server.hasArg("state")) {
    server.send(400, "text/plain", "Missing state");
    return;
  }
  if (server.arg("state") == "1") {
    if (emergencyStopped) {
      server.send(423, "text/plain", "Clear STOP before enabling laser");
      return;
    }
    laserOn = true;
  } else if (server.arg("state") == "0") {
    laserOn = false;
  } else {
    server.send(400, "text/plain", "Invalid state");
    return;
  }
  digitalWrite(LASER_PIN, laserOn ? HIGH : LOW);
  server.send(200, "text/plain", laserOn ? "ON" : "OFF");
}

void handleStop() {
  emergencyStopped = true;
  laserOn = false;
  digitalWrite(LASER_PIN, LOW);
  panTarget = panCurrent;
  tiltTarget = tiltCurrent;
  server.send(200, "text/plain", "Stopped");
}

void handleStatus() {
  String json = "{";
  json += "\"pan\":" + String(panCurrent) + ",";
  json += "\"tilt\":" + String(tiltCurrent) + ",";
  json += "\"laser\":" + String(laserOn ? "true" : "false") + ",";
  json += "\"stopped\":" + String(emergencyStopped ? "true" : "false") + ",";
  json += "\"ip\":\"" + WiFi.localIP().toString() + "\"";
  json += "}";
  server.send(200, "application/json", json);
}

// ==================================================
// SETUP
// ==================================================

void setup() {
  Serial.begin(115200);

  pinMode(LASER_PIN, OUTPUT);
  digitalWrite(LASER_PIN, LOW);

  panServo.setPeriodHertz(50);
  tiltServo.setPeriodHertz(50);

  panServo.attach(PAN_PIN, 600, 2400);
  tiltServo.attach(TILT_PIN, 600, 2400);

  panServo.write(PAN_HOME);
  tiltServo.write(TILT_HOME);

  panCurrent = PAN_HOME;
  tiltCurrent = TILT_HOME;

  panTarget = PAN_HOME;
  tiltTarget = TILT_HOME;

  delay(500);

  // Pure Station Mode (Direct connection to Mobile Hotspot)
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(true);
  WiFi.begin(STA_SSID, STA_PASSWORD);

  Serial.println("\n----------------------------------");
  Serial.println("PoolBot ESP32 Controller (Station Mode)");
  Serial.println("----------------------------------");
  Serial.print("Connecting to Mobile Hotspot: ");
  Serial.println(STA_SSID);

  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 40) {
    delay(500);
    Serial.print(".");
    retries++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Connected! Network IP: http://");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("Warning: Connecting in background...");
  }

  // Setup mDNS responder (http://poolbot.local)
  if (MDNS.begin("poolbot")) {
    Serial.println("mDNS responder started: http://poolbot.local");
  }

  server.on("/", HTTP_GET, handleRoot);
  server.on("/move", HTTP_GET, handleMove);
  server.on("/center", HTTP_GET, handleCenter);
  server.on("/laser", HTTP_GET, handleLaser);
  server.on("/stop", HTTP_GET, handleStop);
  server.on("/status", HTTP_GET, handleStatus);

  server.begin();
  Serial.println("HTTP Web server started.");
}

// ==================================================
// MAIN LOOP
// ==================================================

void loop() {
  server.handleClient();
  updateServos();

  // Automatic background reconnect if hotspot signal drops
  if (WiFi.status() != WL_CONNECTED) {
    static unsigned long lastReconnect = 0;
    if (millis() - lastReconnect > 4000) {
      lastReconnect = millis();
      WiFi.reconnect();
    }
  }
}
