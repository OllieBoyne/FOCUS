import subprocess
import json
from pathlib import Path


def get_frame_count(video_path: str | Path) -> int:
	"""Calculate number of frames in video file."""
	cmd = [
		"ffprobe",
		"-v", "error",
		"-select_streams", "v:0",
		"-count_frames",
		"-show_entries", "stream=nb_read_frames",
		"-of", "json",
		video_path
	]

	result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	info = json.loads(result.stdout)
	return int(info["streams"][0]["nb_read_frames"])