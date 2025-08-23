from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self):
        #19. Initialise a DeepSort Object as "Object_Tracker" property of this class.
        self.object_tracker = DeepSort(
            max_age=20,
            n_init=2,
            nms_max_overlap=0.3,
            max_cosine_distance=0.8,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )

    def track(self, detections, frame):
        #19. Use Update_Tracks function to UPDATE the TRACK from before to this frame. Store in TRACKS.
        #Likely Update Tracks does: Looks at previously sent detection, freshly stored detection, and UPDATES the box while keeping the tracking id the same, 
        #which is essentially, tracking each box!
        tracks = self.object_tracker.update_tracks(detections, frame=frame) 

        print(f"\n--------------------Tracks from DeepSort.update_Tracks(detections, frame) function: --------------------\n")

        tracking_ids = []
        boxes =[]
        #For EACH track (meaning EACH object)
        for track in tracks:
            #Skip non confirmed
            if not track.is_confirmed():
                continue
            tracking_ids.append(track.track_id) #grab tracking ID from THIS track and append to tracking_ids array

            ltrb = track.to_ltrb() #ltrb is left top right bottom
            boxes.append(ltrb)
        #In this ONE Frame, how many boxes, how many tracking_ids? Could be multiple. 
        #So, in this ONE Frame, return all the boxes and their relative tracking ids.
            
        #20. Returns Tracking IDs and Boxes Coordinates that now you can use to draw boxes and list tracking_ids
        return tracking_ids, boxes