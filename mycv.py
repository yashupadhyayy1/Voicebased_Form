import cv2
import face_recognition


def classification():
    # Load the image with the known face
    known_image = face_recognition.load_image_file("AHARNISH.jpg")
    known_face_encoding = face_recognition.face_encodings(known_image)[0]

    # Create arrays of known face encodings and corresponding labels
    known_face_encodings = [known_face_encoding]
    known_face_labels = ["Known Person"]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_labels = []
    process_this_frame = True

    # Start capturing video from the default camera (you can change the index if needed)
    cap = cv2.VideoCapture('WIN_20240227_15_41_28_Pro.mp4')

    while True:
        # Capture each frame from the video feed
        ret, frame = cap.read()

        # Resize the frame to speed up face recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Only process every other frame to speed up face recognition
        if process_this_frame:
            # Find all face locations and face encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            # Initialize an array for face labels
            face_labels = []

            # Loop through each face found in the frame
            for face_encoding in face_encodings:
                # See if the face matches any known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                name = "Unknown"

                # If a match is found, use the label of the known face
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_labels[first_match_index]

                # Add the face label to the array
                face_labels.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_labels):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw the face label below the rectangle
            cv2.putText(frame, name, (left + 6, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return name

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

print(classification())