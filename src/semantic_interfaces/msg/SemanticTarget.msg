uint32 id          # 0 = unknown, database assigns a real ID

int32 track_id     # ByteTrack ID; -1 = no valid track

string class_name
uint16 class_id

float32 confidence

bool has_snapshot  # true only after a successful inspection photo is captured
string snapshot_path  # absolute path to that photo on disk; empty if has_snapshot is false

float32 bbox_left
float32 bbox_top
float32 bbox_right
float32 bbox_bottom

float32 x
float32 y
float32 z