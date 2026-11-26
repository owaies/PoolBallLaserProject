"""
ESP32 Communication and Hardware Controller Module for PoolBot.

Provides a robust HTTP client for interacting with the ESP32 pan/tilt servo assembly.
Enforces all safety rules, configurable soft limits, simulation mode, explicit
movement confirmation, and thread-safe serial command execution.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import requests

logger = logging.getLogger("PoolBot.ESP32Client")


@dataclass
class ServoSoftLimits:
    """Configurable soft limits for pan and tilt servos to protect mechanics."""
    pan_min: int = 15     # Hardware allows 10-180
    pan_max: int = 175
    tilt_min: int = 25    # Hardware dashboard uses 20-160
    tilt_max: int = 155

    def clamp_pan(self, angle: float) -> int:
        val = int(round(angle))
        return max(self.pan_min, min(self.pan_max, val))

    def clamp_tilt(self, angle: float) -> int:
        val = int(round(angle))
        return max(self.tilt_min, min(self.tilt_max, val))

    def is_pan_safe(self, angle: float) -> bool:
        return self.pan_min <= angle <= self.pan_max

    def is_tilt_safe(self, angle: float) -> bool:
        return self.tilt_min <= angle <= self.tilt_max


@dataclass
class ESP32Status:
    """Current state received from the ESP32."""
    connected: bool = False
    pan: int = 90
    tilt: int = 90
    laser: bool = False
    stopped: bool = False
    latency_ms: float = 0.0
    error_message: Optional[str] = None
    is_simulation: bool = False


class ESP32Client:
    """
    Communicates with the ESP32 WebServer over HTTP.

    Safety Enforcements:
    1. Laser is NEVER turned on automatically.
    2. Commands are clamped to configurable soft limits.
    3. Thread-locked to serialize requests and prevent overlapping motion commands.
    4. Simulation mode completely bypasses network calls and simulates hardware.
    5. Network timeouts and errors are caught safely without crashing.
    6. Stop command triggers firmware /stop route and blocks new motion.
    """

    def __init__(
        self,
        base_url: str = "http://192.168.31.2",
        timeout: float = 2.0,
        simulation_mode: bool = True,
        soft_limits: Optional[ServoSoftLimits] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.simulation_mode = simulation_mode
        self.soft_limits = soft_limits or ServoSoftLimits()

        # Thread safety lock
        self._lock = threading.Lock()

        # Local simulated state
        self._sim_pan = 90
        self._sim_tilt = 90
        self._sim_laser = False
        self._sim_stopped = False

        # Last known status
        self.last_status = ESP32Status(is_simulation=self.simulation_mode)

    def set_simulation_mode(self, enabled: bool) -> None:
        """Toggle simulation mode."""
        with self._lock:
            self.simulation_mode = enabled
            logger.info("ESP32Client simulation mode set to: %s", enabled)
            self.last_status.is_simulation = enabled

    def get_status(self) -> ESP32Status:
        """
        Polls the /status endpoint from ESP32 or returns simulated state.
        Response JSON format from firmware:
        {"pan": 90, "tilt": 90, "laser": false, "stopped": false}
        """
        if self.simulation_mode:
            return ESP32Status(
                connected=True,
                pan=self._sim_pan,
                tilt=self._sim_tilt,
                laser=self._sim_laser,
                stopped=self._sim_stopped,
                latency_ms=0.5,
                is_simulation=True,
            )

        start_time = time.time()
        try:
            url = f"{self.base_url}/status"
            resp = requests.get(url, timeout=self.timeout)
            latency = (time.time() - start_time) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                status = ESP32Status(
                    connected=True,
                    pan=int(data.get("pan", 90)),
                    tilt=int(data.get("tilt", 90)),
                    laser=bool(data.get("laser", False)),
                    stopped=bool(data.get("stopped", False)),
                    latency_ms=round(latency, 1),
                    is_simulation=False,
                )
                self.last_status = status
                return status
            else:
                return ESP32Status(
                    connected=False,
                    error_message=f"HTTP {resp.status_code}: {resp.text}",
                    is_simulation=False,
                )
        except requests.RequestException as e:
            return ESP32Status(
                connected=False,
                error_message=f"Connection failed: {str(e)}",
                is_simulation=False,
            )

    def move_servos(
        self,
        pan_angle: float,
        tilt_angle: float,
        require_confirmation: bool = False,
        confirmed: bool = False,
    ) -> Tuple[bool, str]:
        """
        Command pan and tilt servos to target angles.

        Args:
            pan_angle: Target pan in degrees.
            tilt_angle: Target tilt in degrees.
            require_confirmation: If True, moves are rejected unless confirmed=True.
            confirmed: User confirmation token.

        Returns:
            (success, message)
        """
        if require_confirmation and not confirmed:
            return False, "Movement blocked: explicit user confirmation required."

        pan_int = self.soft_limits.clamp_pan(pan_angle)
        tilt_int = self.soft_limits.clamp_tilt(tilt_angle)

        if not self.soft_limits.is_pan_safe(pan_angle):
            logger.warning("Requested pan %.1f° clamped to soft limit %d°", pan_angle, pan_int)
        if not self.soft_limits.is_tilt_safe(tilt_angle):
            logger.warning("Requested tilt %.1f° clamped to soft limit %d°", tilt_angle, tilt_int)

        with self._lock:
            if self.simulation_mode:
                if self._sim_stopped:
                    return False, "Movement stopped. Issue CENTER or clear STOP to resume."
                self._sim_pan = pan_int
                self._sim_tilt = tilt_int
                msg = f"[SIMULATION] Pan set to {pan_int}°, Tilt set to {tilt_int}°"
                logger.info(msg)
                return True, msg

            try:
                # First check status to see if emergency stopped
                status = self.get_status()
                if not status.connected:
                    return False, f"ESP32 disconnected: {status.error_message}"
                if status.stopped:
                    return False, "Movement stopped. ESP32 is in STOP state. Use CENTER to resume."

                # Send pan command: /move?axis=pan&angle=...
                pan_resp = requests.get(
                    f"{self.base_url}/move",
                    params={"axis": "pan", "angle": pan_int},
                    timeout=self.timeout,
                )
                if pan_resp.status_code != 200:
                    return False, f"Failed to set pan: HTTP {pan_resp.status_code} - {pan_resp.text}"

                # Send tilt command: /move?axis=tilt&angle=...
                tilt_resp = requests.get(
                    f"{self.base_url}/move",
                    params={"axis": "tilt", "angle": tilt_int},
                    timeout=self.timeout,
                )
                if tilt_resp.status_code != 200:
                    return False, f"Pan set to {pan_int}°, but tilt failed: HTTP {tilt_resp.status_code} - {tilt_resp.text}"

                return True, f"Servos commanded: Pan {pan_int}°, Tilt {tilt_int}°"

            except requests.RequestException as e:
                return False, f"Network communication error: {str(e)}"

    def center_servos(self) -> Tuple[bool, str]:
        """
        Sends the /center command to ESP32.
        Clears emergency stop and positions both servos at 90°.
        Laser is turned OFF.
        """
        with self._lock:
            if self.simulation_mode:
                self._sim_stopped = False
                self._sim_pan = 90
                self._sim_tilt = 90
                self._sim_laser = False
                return True, "[SIMULATION] Servos centered at 90°, Laser OFF"

            try:
                resp = requests.get(f"{self.base_url}/center", timeout=self.timeout)
                if resp.status_code == 200:
                    return True, "Servos centering command accepted (90°, 90°)"
                return False, f"Center failed: HTTP {resp.status_code} - {resp.text}"
            except requests.RequestException as e:
                return False, f"Network communication error: {str(e)}"

    def stop_servos(self) -> Tuple[bool, str]:
        """
        Sends the /stop command to ESP32.
        Firmware halts movement and disables laser pin immediately.
        NOTE: Servos remain energized holding their current positions.
        This does NOT isolate physical electrical power.
        """
        with self._lock:
            if self.simulation_mode:
                self._sim_stopped = True
                self._sim_laser = False
                return True, "[SIMULATION] EMERGENCY STOP triggered. Movement disabled, Laser OFF."

            try:
                resp = requests.get(f"{self.base_url}/stop", timeout=self.timeout)
                if resp.status_code == 200:
                    return True, "STOPPED: Servo target frozen at current position, Laser OFF."
                return False, f"Stop failed: HTTP {resp.status_code} - {resp.text}"
            except requests.RequestException as e:
                return False, f"Network communication error: {str(e)}"

    def set_laser(self, enable: bool, safety_acknowledgment: bool = False) -> Tuple[bool, str]:
        """
        Explicitly control laser output.
        SAFETY RULE: Laser is NEVER activated automatically.
        Requires manual safety acknowledgment.
        """
        if enable and not safety_acknowledgment:
            return False, "SAFETY BLOCK: Laser activation requires explicit manual safety acknowledgment."

        with self._lock:
            if self.simulation_mode:
                self._sim_laser = enable
                return True, f"[SIMULATION] Laser state set to {'ON' if enable else 'OFF'}"

            try:
                resp = requests.get(
                    f"{self.base_url}/laser",
                    params={"state": "1" if enable else "0"},
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    return True, f"Laser turned {'ON' if enable else 'OFF'}"
                return False, f"Laser command failed: HTTP {resp.status_code} - {resp.text}"
            except requests.RequestException as e:
                return False, f"Network communication error: {str(e)}"
