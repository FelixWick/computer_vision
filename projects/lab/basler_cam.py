import time
from dataclasses import dataclass, replace

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pypylon import genicam, pylon


@dataclass
class CamSettings:
    """All settings BaslerCam.configure() can change. Fields left as None are not touched."""

    exposure_us: float | None = None
    gain_db: float | None = None
    auto_exposure: bool | None = None
    auto_gain: bool | None = None
    auto_white_balance: bool | None = None
    width: int | None = None
    height: int | None = None


class BaslerCam:
    """Small wrapper around a single Basler ace 2 USB camera."""

    def __init__(self, timeout_ms: int = 5000) -> None:
        self.timeout_ms = timeout_ms

        # find available cameras
        factory = pylon.TlFactory.GetInstance()
        devices = factory.EnumerateDevices()
        if len(devices) == 0:
            raise RuntimeError("No Basler camera found.")

        print(f"Found {len(devices)} camera(s):")
        for i, device in enumerate(devices):
            print(f"  [{i}] {device.GetModelName()} (serial: {device.GetSerialNumber()})")

        # take the first camera by default
        device = devices[0]
        self.model = device.GetModelName()
        self.serial = device.GetSerialNumber()
        print(f"Using camera [0]: {self.model} (serial: {self.serial})")

        self.camera = pylon.InstantCamera(factory.CreateDevice(device))
        self.camera.Open()

        # let the camera deliver RGB directly
        self.camera.PixelFormat.Value = "RGB8"

    def get_frame(self) -> np.ndarray:
        """Grab a single frame and return it as an RGB numpy array (height, width, 3).

        Works both on its own and while the camera is grabbing continuously (live feed).
        """
        if self.camera.IsGrabbing():
            grab = self.camera.RetrieveResult(self.timeout_ms, pylon.TimeoutHandling_ThrowException)
        else:
            grab = self.camera.GrabOne(self.timeout_ms)
        try:
            if not grab.GrabSucceeded():
                raise RuntimeError(f"Grab failed: {grab.GetErrorDescription()}")
            return grab.GetArray()
        finally:
            grab.Release()

    def live_feed(self, scale: float = 0.5) -> np.ndarray:
        """Show a live feed in an OpenCV window until it is closed or 'q' / Esc is pressed.

        scale shrinks the displayed image (e.g. 0.5 = half width and height), which keeps
        the display fast on slower machines. The returned frame is always full size.

        Returns the last frame that was grabbed (RGB, full resolution).
        """
        window = f"{self.model} ({self.serial})"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        print("Live feed running, close the window or press 'q' / Esc to stop.")

        # always deliver the newest frame, older ones are dropped
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        try:
            fps = 0.0
            last = time.perf_counter()
            while True:
                frame = self.get_frame()

                # OpenCV expects BGR, and a smaller image is much cheaper to display
                view = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if scale != 1.0:
                    view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                # smoothed frame rate, so the number does not flicker
                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1 / (now - last))
                last = now
                cv2.putText(view, f"{fps:.1f} fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow(window, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.camera.StopGrabbing()
            # unlike destroyWindow, this does not fail if the window was already closed with X
            cv2.destroyAllWindows()
        return frame

    def configure(
        self,
        settings: CamSettings | None = None,
        *,
        exposure_us: float | None = None,
        gain_db: float | None = None,
        auto_exposure: bool | None = None,
        auto_gain: bool | None = None,
        auto_white_balance: bool | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Change the most important camera settings.

        IN MOST CASES CHANGING CAMERA SETTINGS IS NOT NEEDED!!!

        Settings can be given as a CamSettings object, as keyword arguments, or both.
        Keyword arguments override the matching field of the CamSettings object.
        Anything left as None is not touched.

        Setting a manual exposure or gain value switches the matching auto mode off.
        Values outside the camera's range are clamped to the nearest allowed value.
        """
        overrides = {
            "exposure_us": exposure_us,
            "gain_db": gain_db,
            "auto_exposure": auto_exposure,
            "auto_gain": auto_gain,
            "auto_white_balance": auto_white_balance,
            "width": width,
            "height": height,
        }
        s = replace(settings or CamSettings(), **{k: v for k, v in overrides.items() if v is not None})

        if s.auto_exposure is not None:
            self._set("ExposureAuto", "Continuous" if s.auto_exposure else "Off")
        if s.exposure_us is not None:
            self._set("ExposureAuto", "Off")
            self._set("ExposureTime", s.exposure_us)

        if s.auto_gain is not None:
            self._set("GainAuto", "Continuous" if s.auto_gain else "Off")
        if s.gain_db is not None:
            self._set("GainAuto", "Off")
            self._set("Gain", s.gain_db)

        if s.auto_white_balance is not None:
            self._set("BalanceWhiteAuto", "Continuous" if s.auto_white_balance else "Off")

        if s.width is not None:
            self._set("Width", s.width)
        if s.height is not None:
            self._set("Height", s.height)

    def print_settings(self) -> None:
        """Print the current value of the settings handled by configure()."""
        node_map = self.camera.GetNodeMap()
        for name in ("ExposureAuto", "ExposureTime", "GainAuto", "Gain", "BalanceWhiteAuto", "Width", "Height", "PixelFormat"):
            node = node_map.GetNode(name)
            if node is not None and genicam.IsReadable(node):
                print(f"  {name}: {node.ToString()}")

    def close(self) -> None:
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        if self.camera.IsOpen():
            self.camera.Close()

    # allows: with BaslerCam() as cam: ...
    def __enter__(self) -> "BaslerCam":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _set(self, name: str, value) -> None:
        """Set a camera feature by name. Numbers are clamped to the valid range."""
        node = self.camera.GetNodeMap().GetNode(name)
        if node is None or not genicam.IsWritable(node):
            print(f"Setting '{name}' is not available on this camera, skipped.")
            return

        if isinstance(value, str):
            node.Value = value
            return

        value = max(node.Min, min(node.Max, value))
        if isinstance(node.Value, int):
            # integer features (e.g. Width) must be a multiple of their increment
            value = int(node.Min + (value - node.Min) // node.Inc * node.Inc)
        node.Value = value


if __name__ == "__main__":
    # small demo: configure the camera, grab one frame and show it
    with BaslerCam() as cam:
        # settings as a dataclass ...
        settings = CamSettings(exposure_us=10000, gain_db=0.0, auto_white_balance=True)
        cam.configure(settings)

        # ... or as keyword arguments (they can also be combined, keywords win)
        cam.configure(exposure_us=20000)
        print("Current settings:")
        cam.print_settings()

        frame = cam.get_frame()
        print(f"Frame captured: {frame.shape[1]}x{frame.shape[0]}")

        # show the single frame (blocks until the window is closed)
        plt.imshow(frame)
        plt.title(f"{cam.model} ({cam.serial})")
        plt.axis("off")
        plt.show()

        # then start the live feed
        cam.live_feed()
