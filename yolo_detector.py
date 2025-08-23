from ultralytics import YOLO

class YoloDetector:
    #8. Initialise the class, the model path and confidence will be passed when initialising
    def __init__(self, model_path, confidence):
        self.model = YOLO (model_path)
        self.classList = ["person"] #List all the classes here. For now, we are using only person. 
        self.confidence = confidence
    
    #9. The function that labels exactly what is detected in that image (or frame)
    def detect (self, frame):
        results = self.model.predict(frame, conf=self.confidence) #10. Use the Model.Predict(source, confidence) function to get results
        print(f"\n--------------------Results(likely an array?): ${results}--------------------\n")
        #Use the first result 
        result = results[0] #likely an object, meaning results[0] may contain { Boxes: [ box1:[x1,x2,y1,y2], box2:[x1,x2,y1,y2] ], Class: [1,2], Conf: 0.90 }
        print(f"\n--------------------Result in the 0TH Position: ${result}--------------------\n")
        detections = self.make_detections(result) #11. Then, pass the result to a custom "Make Detection" function
        print(f"\n--------------------Detections object: ${detections}--------------------\n")
        return detections #17. Return the array of objects like this: [{[x1, y1, w, h], 1, 0.92}, {}, {}, {}, ..etc]. Back to Yolo_Detection_Tracking
    
    def make_detections(self, result):
        #12. Extract "boxes" attribute (which likely has the coordinates from result object)
        boxes = result.boxes
        print(f"\n--------------------Boxes extracted from result.boxes: ${boxes}--------------------\n")
        detections = [] #Will be an array like this: [{[x1, y1, w, h], 1, 0.92}, {}, {}, {}, ..etc]
        #13. For each "box" attribute in boxes, extract x, y coordinates
        for box in boxes:
            print(f"Each individual box: ${box}")
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w,h = x2 - x1, y2 - y1 #calculate width and height based on coordinates for that particular box.
            class_number = int(box.cls[0])
            print(f"\n--------------------Each box's class_number- within box.cls attribute: ${class_number}--------------------\n")
            print(f"\n--------------------Box's class_number used in result.names[class_number]. Should be the name: ${result.names[class_number]}--------------------\n")
            #14. Of THAT box's result, extract the class name using result.names[class_number] and check if it is in our interested list of classes. If not, break loop and continue. 
            if result.names[class_number] not in self.classList:
                continue

            #15. Extract the confidence of that class prediction as well.
            box_confidence = box.conf[0]
            print(f"\n--------------------THAT box's confidence score: ${box_confidence}--------------------\n")
            #16. Push to array an object: (BOXCOORDINATES, CLASS, CONFIDENCE )
            detections.append( ( ([x1, y1, w, h]), class_number, box_confidence ) )
        return detections