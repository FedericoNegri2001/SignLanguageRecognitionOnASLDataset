1)OPTIONAL create a virtual enviroment .conda
2)pip install -r requirements.txt


IF YOU WANT TO USE THE ASL DATASET-->EXECUTE ONLY realtime_detection.py with RandomForestClassifier's weight "model_rf_90" 


IF YOU WANT TO USE ANOTHER DATASET-->EXECUTE IN ORDER-->create_landmarks.py train_Random_Forest.py realtime_detection.py     
P.S. import your dataset in the project folder and use "file.pickle" and "weights.p" in train_Random_Forest.py and realtime_detection.py respectively!
