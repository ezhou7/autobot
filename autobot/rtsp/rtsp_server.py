import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')

from gi.repository import GLib, Gst, GstRtspServer
from autobot.rtsp import get_wlan0_ip
from autobot.utils.logging import stdout_logger


logger = stdout_logger(__name__)


class RTSPMediaFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, device_name, **props):
        super().__init__(**props)       
        self.launch_string = (
           f"v4l2src device={device_name} ! " 
            "image/jpeg,width=1920,height=1080,framerate=30/1 ! "
            "jpegdec ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=ultrafast ! "
            "rtph264pay name=pay0 pt=96 config-interval=1"
        )

    def do_create_element(self, url):
        logger.info("Creating pipeline for: ", url)
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        logger.info("Media configured: ", rtsp_media)


class AutobotRTSPServer:
    def __init__(self, device_name: str = '/dev/video0', ip_add: str = None):
        Gst.init(None)

        self.ip_add = ip_add
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service("8554")
        self.server.set_address("0.0.0.0")
        
        factory = RTSPMediaFactory(device_name=device_name)
        factory.set_shared(True)
        
        mount_points = self.server.get_mount_points()
        mount_points.add_factory("/test", factory)
        
        self.server.attach(None)
        logger.info(f"RTSP Server is running at rtsp://{self.ip_add}:8554/test")


if __name__ == "__main__":
    ip_addr = get_wlan0_ip("wlx90de8012ad37")
    # ip_addr = "127.0.0.1"
    server = AutobotRTSPServer(device_name="/dev/video20", ip_add=ip_addr)
    loop = GLib.MainLoop()
    try:
        loop.run()
    except:
        pass
