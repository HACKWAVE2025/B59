# AuthentiHire
## AI-Powered Deepfake & Authenticity Detection for Smarter Recruitment
**AuthentiHire** is an AI-based web platform designed to ensure fair, secure, and authentic remote hiring.
It automatically detects deepfakes, gaze inconsistencies, AI-generated audio, bias, and emotional behavior, ensuring that interview candidates are genuine and fairly evaluated.
# Features
## Deepfake & Video Authenticity Detection
Detects AI-generated or manipulated faces in real-time using DeepFace and other deepfake detection models.
Performs liveness checks to ensure the candidate is a real, live person — not a replayed video.
## Gaze Tracking & Focus Monitoring
Tracks eye gaze and focus direction to detect whether the candidate is attentive or looking away.
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
Layer	Technologies
Frontend	HTML, CSS, JavaScript
Backend	Flask (Python)
Database	SQLite
AI/ML Libraries	DeepFace, FER, OpenCV, NumPy
Real-Time Communication	Flask-SocketIO
Other Tools	Threading, JSON, WebRTC
### Workflow
Candidate joins an interview session.
Video and audio streams are captured in real time.
Flask backend performs multiple analyses:
Deepfake face detection
Emotion and gaze tracking
Audio authenticity check
Liveness verification
All results are stored in an SQLite database.
A dashboard shows real-time insights such as:
Deepfake confidence score
Emotion trends
Gaze focus
Bias detection outcomes
Liveness percentage
# Database Schema (SQLite)
Database: candidates.db


## Example Dashboard Insights
Metric	Example Output
Deepfake Confidence	97.8% Real
Emotion Summary	45% Neutral, 35% Happy, 20% Surprised
Gaze Stability	90% Focused
Liveness Score	96%
Bias Detection	No major bias detected
## Ethics & Privacy
AuthentiHire follows strict ethical AI practices:
No data sharing or third-party storage.
Transparent and explainable AI outputs.
Focus on fairness and unbiased evaluation.
## Future Enhancements
Integration with Zoom/Google Meet interview APIs.
<br>
Improved audio deepfake classifier.
Recruiter bias visualization dashboard.
Multilingual emotion recognition.
