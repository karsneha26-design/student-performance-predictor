from django.shortcuts import render
import shap
import numpy as np 
from .models import PredictionHistory
import joblib
import os
from django.http import JsonResponse

# Load the trained ML model
model_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'student_performance_model.pkl'
)

model = joblib.load(model_path)


def home(request):
    prediction = None
    what_if_prediction = None
    risk_level = None
    recommendations = []
    feature_importance = None
    performance_width = None

    study_hours = ''
    attendance = ''
    previous_marks = ''
    assignments = ''

    if request.method == 'POST':

        study_hours = request.POST.get('study_hours', '')
        attendance = request.POST.get('attendance', '')
        previous_marks = request.POST.get('previous_marks', '')
        assignments = request.POST.get('assignments', '')

        # What-If Analysis
        if 'what_if_hours' in request.POST:

            what_if_hours = request.POST.get('what_if_hours', '')

            result = model.predict([[
                float(what_if_hours),
                float(attendance),
                float(previous_marks),
                float(assignments)
            ]])[0]

            what_if_prediction = round(result, 2)

        else:

            # Make normal prediction
            result = model.predict([[
                float(study_hours),
                float(attendance),
                float(previous_marks),
                float(assignments)
            ]])[0]

            prediction = round(result, 2)
            performance_width = max(0, min(100, prediction))
            request.session['prediction'] = prediction
            request.session['attendance'] = attendance
            request.session['study_hours'] = study_hours
            request.session['previous_marks'] = previous_marks
            request.session['assignments'] = assignments
            request.session['recommendations'] = recommendations
            
            # AI Model Explainability
            feature_names = [
                "Study Hours",
                "Attendance",
                "Previous Marks",
                "Assignments"
            ]

            explainer = shap.TreeExplainer(model)

            student_data = np.array([[
                float(study_hours),
                float(attendance),
                float(previous_marks),
                float(assignments),
            ]])

            shap_values = explainer(student_data)

            values = shap_values.values[0]

            total = sum(abs(value) for value in values)

            if total == 0:
                feature_importance = {
                    name: 0 for name in feature_names
                }
            else:
                feature_importance = dict(
                    zip(
                        feature_names,
                        [
                             round(abs(value) / total * 100, 1)
                             for value in values
                        ]
                    )
                )


            # Determine student risk level
            if prediction >= 70:
                risk_level = "Low Risk"
            elif prediction >= 50:
                risk_level = "Moderate Risk"
            else:
                risk_level = "High Risk"
            request.session['risk_level'] = risk_level 

            PredictionHistory.objects.create(
                study_hours=study_hours,
                attendance=attendance,
                previous_marks=previous_marks,
                assignments=assignments,
                predicted_performance=prediction,
                risk_level=risk_level
            )

            if risk_level == "Low Risk":
                ai_insight = "Your predicted performance is strong. Keep maintaining your current study habits."
            elif risk_level == "Moderate Risk":
                ai_insight = "Your performance has room for improvement. Focus on consistency and regular practice."
            else:
                ai_insight = "Your current performance needs attention. Increasing study time and improving attendance may help."

            request.session['ai_insight'] = ai_insight   

            # Generate personalized recommendations
            if float(study_hours) < 4:
                recommendations.append(
                    "Try increasing your daily study hours."
                )

            if float(attendance) < 75:
                recommendations.append(
                    "Improve your attendance for more consistent learning."
                )

            if float(previous_marks) < 60:
                recommendations.append(
                    "Spend more time revising and practicing previous topics."
                )

            if float(assignments) < 6:
                recommendations.append(
                    "Complete more assignments regularly to strengthen your preparation."
                )

            if not recommendations:
                recommendations.append(
                    "Your current study habits look good. Keep maintaining your consistency!"
                )

    return render(request, 'home.html', {
        'prediction': prediction,
        'what_if_prediction': what_if_prediction,
        'risk_level': risk_level,
        'recommendations': recommendations,
        'feature_importance': feature_importance,
        'performance_width': performance_width,
        'study_hours': study_hours,
        'attendance': attendance,
        'previous_marks': previous_marks,
        'assignments': assignments,
        'risk_level': risk_level
    })

def analytics(request):
    prediction = request.session.get('prediction')
    attendance = request.session.get('attendance')
    study_hours = request.session.get('study_hours')
    previous_marks = request.session.get('previous_marks')
    assignments = request.session.get('assignments')
    risk_level = request.session.get('risk_level')
    ai_insight = request.session.get('ai_insight')
    recommendations = request.session.get('recommendations', [])

    history = PredictionHistory.objects.order_by('-created_at')[:5]

    return render(request, 'analytics.html', {
        'prediction': prediction,
        'attendance': attendance,
        'study_hours': study_hours,
        'previous_marks': previous_marks,
        'assignments': assignments,
        'risk_level': risk_level,
        'ai_insight': ai_insight,
        'recommendations': recommendations,
        'history': history
    })