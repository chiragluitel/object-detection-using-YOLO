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
        #Store tuple: IDs, boxes and Labels
        tracking_results = []
        #For EACH track (meaning EACH object)
        for det, track in zip(detections, tracks):
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            class_number = det[1]
            label = self.get_label_from_detection(class_number)
            ltrb = track.to_ltrb()
            tracking_results.append((track_id, ltrb, label))
        #In this ONE Frame, how many boxes, how many tracking_ids? Could be multiple. 
        #So, in this ONE Frame, return all the boxes and their relative tracking ids.
            
        #20. Returns Tracking IDs and Boxes Coordinates that now you can use to draw boxes and list tracking_ids
        return tracking_results
    
    def get_label_from_detection(self, class_number):
    # Map class_number to class name
        class_names = ["person", "cell phone", "bottle", "cup", "book"]  # Add more classes if needed
        if class_number < len(class_names):
            return class_names[class_number]
        return "Unknown"
