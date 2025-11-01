# AuthentiHire
## AI-Powered Deepfake & Authenticity Detection for Smarter Recruitment
**AuthentiHire** is an AI-based web platform designed to ensure fair, secure, and authentic remote hiring.<br>
It automatically detects deepfakes, gaze inconsistencies, AI-generated audio, bias, and emotional behavior, ensuring that interview candidates are genuine and fairly evaluated.
# Features
## Deepfake & Video Authenticity Detection
Detects AI-generated or manipulated faces in real-time using DeepFace and other deepfake detection models.<br>
Performs liveness checks to ensure the candidate is a real, live person — not a replayed video.
## Gaze Tracking & Focus Monitoring
Tracks eye gaze and focus direction to detect whether the candidate is attentive or looking away.<br>
Flags suspicious gaze shifts (e.g., reading from a script).
## Real vs AI Human Detection
Uses facial and motion analysis to verify human presence vs AI-generated avatar.
## Audio Deepfake Detection
Analyzes candidate audio to detect AI-generated or modified speech patterns.
## Emotion & Expression Analysis
Detects emotions such as happy, sad, angry, surprised, neutral, etc., using DeepFace and FER libraries.
## Multi-Person Detection
Ensures only one candidate is present by detecting multiple faces in the frame.
## Bias Detection
Checks interview analytics for potential bias in decision-making or data interpretation.
## Liveness & Behavioral Check
Observes micro-movements, natural blinking, and head motion to confirm real human activity.
# Tech Stack
Layer	Technologies<br>
Frontend	HTML, CSS, JavaScript<br>
Backend	Flask (Python)<br>
Database	SQLite<br>
AI/ML Libraries	DeepFace, FER, OpenCV, NumPy<br>
Real-Time Communication	Flask-SocketIO<br>
Other Tools	Threading, JSON, WebRTC<br>
### Workflow
Candidate joins an interview session.<br>
Video and audio streams are captured in real time.<br>
Flask backend performs multiple analyses:<br>
Deepfake face detection<br>
Emotion and gaze tracking<br>
Audio authenticity check<br>
Liveness verification<br>
All results are stored in an SQLite database.<br>
A dashboard shows real-time insights such as:<br>
Deepfake confidence score<br>
Emotion trends<br>
Gaze focus<br>
Bias detection outcomes<br>
Liveness percentage<br>
# Database Schema (SQLite)
Database: candidates.db<br>


## Example Dashboard Insights
Metric	Example Output<br>
Deepfake Confidence	97.8% Real<br>
Emotion Summary	45% Neutral, 35% Happy, 20% Surprised<br>
Gaze Stability	90% Focused<br>
Liveness Score	96%<br>
Bias Detection	No major bias detected<br>
## Ethics & Privacy
AuthentiHire follows strict ethical AI practices:<br>
No data sharing or third-party storage.<br>
Transparent and explainable AI outputs.<br>
Focus on fairness and unbiased evaluation.<br>
## Future Enhancements
Integration with Zoom/Google Meet interview APIs.<br>
<br>
Improved audio deepfake classifier.<br>
Recruiter bias visualization dashboard.<br>
Multilingual emotion recognition.<br>
