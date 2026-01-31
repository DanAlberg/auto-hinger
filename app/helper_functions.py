from ppadb.client import Client as AdbClient
import subprocess
import time


def ensure_adb_running():
    """Ensure that the Android Debug Bridge (ADB) server is running."""
    print("Checking ADB status...")
    try:
        result = subprocess.run(["adb", "get-state"], capture_output=True, text=True)
        if result.returncode != 0 or "device" not in result.stdout.lower():
            print("ADB not running, starting server...")
            subprocess.run(["adb", "start-server"], check=True)
            print("ADB started successfully")
        else:
            print("ADB is already running")
    except FileNotFoundError:
        raise RuntimeError("ADB not found. Please install Android Platform Tools and add to PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to start ADB: {e}")


def connect_device(user_ip_address: str = "127.0.0.1"):
    adb = AdbClient(host=user_ip_address, port=5037)
    devices = adb.devices()
    if len(devices) == 0:
        print("No devices connected")
        return None
    device = devices[0]
    print(f"Connected to {device.serial}")
    return device


def tap(device, x: int, y: int) -> None:
    device.shell(f"input tap {x} {y}")


def swipe(device, x1: int, y1: int, x2: int, y2: int, duration: int = 500) -> None:
    device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration}")


def get_screen_resolution(device):
    output = device.shell("wm size")
    resolution = output.strip().split(":")[1].strip()
    width, height = map(int, resolution.split("x"))
    return width, height


def open_hinge(device) -> None:
    device.shell("monkey -p co.hinge.app 1")
    time.sleep(1)


def reset_hinge_app(device) -> None:
    device.shell("am force-stop co.hinge.app")
    time.sleep(0.5)
    open_hinge(device)
