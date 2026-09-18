from django.db import models

class PredictionHistory(models.Model):
    study_hours = models.FloatField()
    attendance = models.FloatField()
    previous_marks = models.FloatField()
    assignments = models.FloatField()
    predicted_performance = models.FloatField()
    risk_level = models.CharField(max_length=30)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_performance}% - {self.risk_level}"

# Create your models here.
