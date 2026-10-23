import matplotlib.pyplot as plt
from pypylon import pylon

# find available cameras
factory = pylon.TlFactory.GetInstance()
devices = factory.EnumerateDevices()

if len(devices) == 0:
    raise SystemExit("No Basler camera found.")

print(f"Found {len(devices)} camera(s):")
for i, device in enumerate(devices):
    print(f"  [{i}] {device.GetModelName()} (serial: {device.GetSerialNumber()})")

# take the first camera by default
device = devices[0]
print(f"Using camera [0]: {device.GetModelName()} (serial: {device.GetSerialNumber()})")

camera = pylon.InstantCamera(factory.CreateDevice(device))
camera.Open()

# let the camera deliver RGB directly
camera.PixelFormat.Value = "RGB8"

# grab a single frame (timeout in ms)
grab = camera.GrabOne(5000)
if not grab.GrabSucceeded():
    camera.Close()
    raise SystemExit(f"Grab failed: {grab.GetErrorDescription()}")

frame = grab.GetArray()  # RGB numpy array (height, width, 3)

grab.Release()
camera.Close()

print(f"Frame captured: {frame.shape[1]}x{frame.shape[0]}")

# display the frame (blocks until the window is closed)
plt.imshow(frame)
plt.title(f"{device.GetModelName()} ({device.GetSerialNumber()})")
plt.axis("off")
plt.show()
