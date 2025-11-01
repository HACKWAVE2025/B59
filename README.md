**AuthentiHire**
**AI-Powered Deepfake & Authenticity Detection for Smarter Recruitment**
AuthentiHire is an AI-based web platform designed to ensure fair, secure, and authentic remote hiring. It automatically detects deepfakes, gaze inconsistencies, AI-generated audio, bias, and emotional behavior, ensuring that interview candidates are genuine and unbiasedly evaluated. Features
 Video Authenticity & Deepfake Detection
Detects AI-generated or tampered faces in real-time using DeepFace and deepfake analysis models.
Performs liveness checks to ensure that a real person (not a replayed video) is being recorded.
Gaze Tracking & Focus Monitoring
Monitors eye movements and focus direction to detect whether the candidate is attentive, reading a script, or looking away frequently.
Real vs AI Human Detection
Uses motion, micro-expression, and frame-based AI analysis to verify human presence vs AI-generated avatars.
Audio Deepfake Detection
Identifies synthetic voices or manipulated audio to flag suspicious or AI-altered speech.
Emotion & Expression Analysis
Employs DeepFace emotion recognition to classify emotional states such as happiness, anger, surprise, or fear during interviews.
 Multi-Person Detection
Detects multiple faces in the frame to ensure no one else is present or prompting the candidate.
 Bias Detection
Monitors interview responses and analysis results for potential algorithmic or interviewer bias to promote fair hiring.
 Liveness & Behavior Check
Tracks natural rhythm, micro-movements, and blinking patterns to confirm real-time human presence.
 Tech Stack
Layer	Technologies
Frontend	HTML, CSS, JavaScript
Backend	Flask (Python)
Database	SQLite
AI/ML Libraries	DeepFace, FER (Facial Emotion Recognition), OpenCV, NumPy
Communication	Flask-SocketIO for real-time video/audio streaming
Other Tools	Threading, CV2, JSON, WebRTC (for live video feed)
 System Workflow
User joins the interview (via web interface).
Video & Audio streams are captured in real-time.
Flask backend performs:
Deepfake image analysis
Audio authenticity detection
Emotion recognition and gaze tracking
Results are logged in the SQLite database for further review.
The dashboard displays analytics such as:
Emotion timeline
Gaze focus report
Liveness score
Deepfake confidence levels
Bias or irregularity flags
 Database Structure (SQLite)
Database Name: candidates.db
Tables:
gaze_data(meeting_id, timestamp, x, y, focus_state)
emotion_data(meeting_id, frame_id, emotion, confidence)
deepfake_results(meeting_id, frame_id, score, label)
audio_results(meeting_id, timestamp, score, label)
participants(id, name, join_time)
