# AI-Powered-Disease-Detection-with-Photos-and-Symptom-Analysis
#Hackstreet Boys

Project Demo video:https://github-production-user-asset-6210df.s3.amazonaws.com/201115426/573503357-2983c720-4e6c-4caa-a53a-ce2d51260c15.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20260403%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260403T214650Z&X-Amz-Expires=300&X-Amz-Signature=57c1b8d16eefe79cbfe6337a560b406113f09bad819d4b7dae218b1b5a8bfe4a&X-Amz-SignedHeaders=host

Successfully deployed my AI model, and it’s performing well. Looking forward to adding more improvements and new features soon.
check it out:https://whyhealth.streamlit.app/


Health AI Diagnosis and Patient Monitoring System

This project is an AI-powered healthcare assistance system developed using Streamlit, PyTorch, EfficientNet, and BioBERT. It is designed to support early-level health screening and patient follow-up by combining image-based disease detection with symptom-based analysis. The system accepts eye images, tongue images, skin images, and symptom text input, and then provides a final disease prediction along with a confidence score.

The application includes image classification models for three different health indicators. The eye module is capable of identifying conditions such as cataract, thyroid-related eye signs, uveitis, watery eyes caused by fever, and healthy eyes. The tongue module analyzes tongue images to detect conditions like oral cancer, diabetes signs, prediabetes signs, and healthy tongue status. The skin module allows users to upload skin images from the computer and classifies multiple skin-related conditions including melanoma, benign keratosis, atopic dermatitis, ringworm or candidiasis, squamous cell carcinoma, and vascular lesions.

In addition to image analysis, the project also includes a symptom-based disease prediction system powered by BioBERT. Users can enter symptoms such as fever, headache, weakness, body pain, fatigue, vomiting, and breathing issues. The symptom text is preprocessed and passed through a fine-tuned transformer model to predict the most likely disease. The final system combines visual observations from eye, tongue, and skin analysis with the symptom-based prediction to generate a more context-aware disease diagnosis.

A key feature of this project is patient history tracking. Each patient is assigned a unique patient ID, and every new consultation saves the uploaded or captured images into a dedicated history folder. This allows users or healthcare professionals to view previously saved eye, tongue, and skin images whenever the history option is selected. The system supports comparison of current and previous images using feature embeddings extracted from EfficientNet, helping identify whether the condition is stable or if noticeable visual changes have occurred over time.

The project also includes a custom reminder and notification system where users can schedule reminders for medicine, hydration, sugar checks, or follow-up consultations. These reminders are stored with custom date and time values and can be used as a personal health monitoring assistant.

This project is suitable for healthcare innovation projects, hackathons, remote diagnosis systems, and patient follow-up dashboards. It demonstrates the integration of computer vision, natural language processing, and frontend application development into a single healthcare-focused AI solution. Future improvements can include cloud-based storage, doctor dashboards, PDF medical reports, appointment scheduling, mobile application integration, and anomaly detection techniques such as PaDiM for advanced longitudinal comparison.
